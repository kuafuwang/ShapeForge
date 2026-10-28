// Independent Go verifier for the Phase 2C1 cross-language evidence golden.
// It uses only the Go standard library and never imports or executes Python.
package main

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math/big"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

type Golden struct {
	Blob                  *BlobGolden     `json:"npu_blob"`
	Bundle                *BundleGolden   `json:"npu_bundle"`
	Build                 *BuildGolden    `json:"npu_build"`
	SourceBase64          string          `json:"source_base64"`
	Request               map[string]any  `json:"request"`
	RequestDigestDocument map[string]any  `json:"request_digest_document"`
	RequestCanonicalJSON  string          `json:"request_canonical_json"`
	RequestDigest         string          `json:"request_digest"`
	Receipt               ReceiptGolden   `json:"receipt"`
	Signature             SignatureGolden `json:"signature"`
	Timeouts              []TimeoutGolden `json:"timeout_seconds_to_ms"`
	RejectedTimeouts      []string        `json:"timeout_seconds_rejected"`
}

// BlobGolden freezes the P0b blob-reference contract documents.  It is verified
// independently from the Python implementation: the Go side rebuilds the
// canonical request, receipt and signed result from the raw fields.
type BlobGolden struct {
	Request               map[string]any  `json:"request"`
	RequestDigestDocument map[string]any  `json:"request_digest_document"`
	RequestCanonicalJSON  string          `json:"request_canonical_json"`
	RequestDigest         string          `json:"request_digest"`
	Receipt               ReceiptGolden   `json:"receipt"`
	Signature             SignatureGolden `json:"signature"`
}

// BundleGolden freezes the P1 bundle-contract documents.  The Go side rebuilds
// the canonical manifest, the artifact-set digest, the request document, the
// receipt and the signed result from the raw fields.
type BundleGolden struct {
	Request               map[string]any   `json:"request"`
	BundleManifest        map[string]any   `json:"bundle_manifest"`
	ArtifactSetDigest     string           `json:"artifact_set_digest"`
	RequestDigestDocument map[string]any   `json:"request_digest_document"`
	RequestCanonicalJSON  string           `json:"request_canonical_json"`
	RequestDigest         string           `json:"request_digest"`
	Receipt               ReceiptGolden    `json:"receipt"`
	Evidence              map[string]any   `json:"evidence"`
	CompletenessVectors   []EvidenceVector `json:"completeness_vectors"`
	Signature             SignatureGolden  `json:"signature"`
}

// EvidenceVector is one TimingEvidence document plus the verdict the Python
// implementation reached for it.  Go re-derives the verdict from the declared
// topology instead of trusting the flag, so the device/rank/die coverage rule
// cannot drift between the two languages.
type EvidenceVector struct {
	Name     string         `json:"name"`
	Valid    bool           `json:"valid"`
	Document map[string]any `json:"document"`
}

type ReceiptGolden struct {
	RequestID     string         `json:"request_id"`
	AttemptID     string         `json:"attempt_id"`
	Document      map[string]any `json:"document"`
	CanonicalJSON string         `json:"canonical_json"`
	ReceiptID     string         `json:"receipt_id"`
}

type SignatureGolden struct {
	KeyID          string         `json:"key_id"`
	KeyHex         string         `json:"key_hex"`
	ReportBase64   string         `json:"report_base64"`
	EvidenceBase64 string         `json:"evidence_base64"`
	UnsignedResult map[string]any `json:"unsigned_result"`
	Document       map[string]any `json:"document"`
	CanonicalJSON  string         `json:"canonical_json"`
	HMACSHA256     string         `json:"hmac_sha256"`
}

type TimeoutGolden struct {
	Seconds      string `json:"seconds"`
	Milliseconds int64  `json:"milliseconds"`
}

var requestFields = []string{
	"kind", "run_id", "session_id", "task", "dtype", "size", "target_id",
	"task_fingerprint", "candidate_sha256", "mode", "seed", "reps",
	"queue_timeout_ms", "candidate_timeout_ms", "cases", "fast_bench",
	"require_contract_launch", "validate", "measure_serial", "cache_policy",
}

func shaHex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// appendJSONString implements the string rules needed by RFC 8785 directly.
// Unlike encoding/json it does not HTML-escape <, >, &, U+2028, or U+2029.
func appendJSONString(buf *bytes.Buffer, value string) error {
	if !utf8.ValidString(value) {
		return errors.New("invalid UTF-8 string")
	}
	const hexDigits = "0123456789abcdef"
	buf.WriteByte('"')
	for _, r := range value {
		switch r {
		case '"', '\\':
			buf.WriteByte('\\')
			buf.WriteRune(r)
		case '\b':
			buf.WriteString("\\b")
		case '\f':
			buf.WriteString("\\f")
		case '\n':
			buf.WriteString("\\n")
		case '\r':
			buf.WriteString("\\r")
		case '\t':
			buf.WriteString("\\t")
		default:
			if r < 0x20 {
				buf.WriteString("\\u00")
				buf.WriteByte(hexDigits[(r>>4)&0xf])
				buf.WriteByte(hexDigits[r&0xf])
			} else {
				buf.WriteRune(r)
			}
		}
	}
	buf.WriteByte('"')
	return nil
}

func appendCanonical(buf *bytes.Buffer, value any, depth int) error {
	if depth > 32 {
		return errors.New("canonical document nesting is too deep")
	}
	switch typed := value.(type) {
	case nil:
		buf.WriteString("null")
		return nil
	case bool:
		if typed {
			buf.WriteString("true")
		} else {
			buf.WriteString("false")
		}
		return nil
	case string:
		return appendJSONString(buf, typed)
	case int:
		buf.WriteString(strconv.Itoa(typed))
		return nil
	case int64:
		buf.WriteString(strconv.FormatInt(typed, 10))
		return nil
	case json.Number:
		text := typed.String()
		integer, ok := new(big.Int).SetString(text, 10)
		if !ok || strings.ContainsAny(text, ".eE") || strings.HasPrefix(text, "+") {
			return fmt.Errorf("non-integer canonical number %q", text)
		}
		limit := new(big.Int).SetInt64(9007199254740991)
		if new(big.Int).Abs(integer).Cmp(limit) > 0 {
			return fmt.Errorf("unsafe canonical integer %q", text)
		}
		buf.WriteString(text)
		return nil
	case []any:
		buf.WriteByte('[')
		for index, item := range typed {
			if index != 0 {
				buf.WriteByte(',')
			}
			if err := appendCanonical(buf, item, depth+1); err != nil {
				return err
			}
		}
		buf.WriteByte(']')
		return nil
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			if key == "" || !isASCII(key) {
				return fmt.Errorf("canonical key is empty or non-ASCII: %q", key)
			}
			keys = append(keys, key)
		}
		sort.Strings(keys)
		buf.WriteByte('{')
		for index, key := range keys {
			if index != 0 {
				buf.WriteByte(',')
			}
			if err := appendJSONString(buf, key); err != nil {
				return err
			}
			buf.WriteByte(':')
			if err := appendCanonical(buf, typed[key], depth+1); err != nil {
				return err
			}
		}
		buf.WriteByte('}')
		return nil
	default:
		return fmt.Errorf("unsupported canonical value %T", value)
	}
}

func isASCII(value string) bool {
	for _, r := range value {
		if r > 127 {
			return false
		}
	}
	return true
}

func canonical(value any) ([]byte, error) {
	var buf bytes.Buffer
	if err := appendCanonical(&buf, value, 0); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func requestDocument(request map[string]any) (map[string]any, error) {
	if len(request) != len(requestFields) {
		return nil, fmt.Errorf("request has %d fields, want %d", len(request), len(requestFields))
	}
	canonicalRequest := make(map[string]any, len(requestFields))
	for _, name := range requestFields {
		value, present := request[name]
		if !present {
			return nil, fmt.Errorf("request field %s is missing", name)
		}
		canonicalRequest[name] = value
	}
	return map[string]any{
		"document_type":    "tcenv.remote.v1.request-digest",
		"document_version": 1,
		"digest_algorithm": "sha256-rfc8785-v1",
		"request":          canonicalRequest,
	}, nil
}

var blobRequestFields = []string{
	"kind", "run_id", "session_id", "task", "dtype", "size", "target_id",
	"task_fingerprint", "candidate_digest", "execution", "blob_ref", "mode", "seed", "reps",
	"queue_timeout_ms", "candidate_timeout_ms", "cases", "fast_bench",
	"require_contract_launch", "validate", "measure_serial", "cache_policy",
}

func blobRequestDocument(request map[string]any) (map[string]any, error) {
	if len(request) != len(blobRequestFields) {
		return nil, fmt.Errorf("blob request has %d fields, want %d", len(request), len(blobRequestFields))
	}
	canonicalRequest := make(map[string]any, len(blobRequestFields))
	for _, name := range blobRequestFields {
		value, present := request[name]
		if !present {
			return nil, fmt.Errorf("blob request field %s is missing", name)
		}
		canonicalRequest[name] = value
	}
	blob, ok := canonicalRequest["blob_ref"].(map[string]any)
	if !ok {
		return nil, errors.New("blob_ref is not an object")
	}
	candidate, ok := canonicalRequest["candidate_digest"].(map[string]any)
	if !ok {
		return nil, errors.New("candidate_digest is not an object")
	}
	if candidate["algorithm"] != "sha256-blob-v1" || candidate["value"] != blob["sha256"] {
		return nil, errors.New("candidate digest does not bind the blob descriptor")
	}
	if blob["content_id"] != "blob-sha256-v1:"+fmt.Sprint(blob["sha256"]) {
		return nil, errors.New("content id does not bind the blob digest")
	}
	execution, ok := canonicalRequest["execution"].(map[string]any)
	if !ok {
		return nil, errors.New("execution is not an object")
	}
	if execution["source_media_type"] != blob["media_type"] {
		return nil, errors.New("blob media type differs from the execution media type")
	}
	return map[string]any{
		"document_type":    "tcenv.remote.npu-blob.request-digest",
		"document_version": 3,
		"digest_algorithm": "sha256-rfc8785-v1",
		"request":          canonicalRequest,
	}, nil
}

func blobReceiptDocument(requestID, attemptID, requestDigest string, candidate, group any) map[string]any {
	return map[string]any{
		"document_type":    "tcenv.remote.npu-blob.receipt-id",
		"document_version": 3,
		"digest_algorithm": "sha256-rfc8785-v1",
		"request_id":       requestID,
		"attempt_id":       attemptID,
		"request_digest":   requestDigest,
		"candidate_digest": candidate,
		"device_group":     group,
	}
}

func blobResultDocument(unsigned map[string]any, keyID string) (map[string]any, error) {
	return map[string]any{
		"document_type":       "tcenv.remote.npu-blob.result-signature",
		"document_version":    3,
		"signature_algorithm": "hmac-sha256-v1",
		"key_id":              keyID,
		"provenance":          unsigned["provenance"],
		"report":              unsigned["report"],
		"evidence":            unsigned["evidence"],
		"receipt":             unsigned["receipt"],
	}, nil
}

func receiptDocument(requestID, attemptID, requestDigest string) map[string]any {
	return map[string]any{
		"document_type":    "tcenv.remote.v1.receipt-id",
		"document_version": 1,
		"digest_algorithm": "sha256-rfc8785-v1",
		"request_id":       requestID,
		"attempt_id":       attemptID,
		"request_digest":   requestDigest,
	}
}

func resultDocument(unsigned map[string]any, keyID string) (map[string]any, error) {
	if len(unsigned) != 4 {
		return nil, fmt.Errorf("unsigned result has %d fields, want 4", len(unsigned))
	}
	for _, name := range []string{"provenance", "report", "evidence", "receipt"} {
		if _, present := unsigned[name]; !present {
			return nil, fmt.Errorf("unsigned result field %s is missing", name)
		}
	}
	return map[string]any{
		"document_type":       "tcenv.remote.v1.result-signature",
		"document_version":    1,
		"signature_algorithm": "hmac-sha256-v1",
		"key_id":              keyID,
		"provenance":          unsigned["provenance"],
		"report":              unsigned["report"],
		"evidence":            unsigned["evidence"],
		"receipt":             unsigned["receipt"],
	}, nil
}

func checkDocument(actual map[string]any, expected map[string]any, expectedJSON, expectedDigest, name string) error {
	actualBytes, err := canonical(actual)
	if err != nil {
		return fmt.Errorf("%s canonicalization: %w", name, err)
	}
	expectedBytes, err := canonical(expected)
	if err != nil {
		return fmt.Errorf("%s expected document: %w", name, err)
	}
	if !bytes.Equal(actualBytes, expectedBytes) {
		return fmt.Errorf("%s independently constructed document differs", name)
	}
	if string(actualBytes) != expectedJSON {
		return fmt.Errorf("%s canonical JSON differs\n got: %s\nwant: %s", name, actualBytes, expectedJSON)
	}
	if digest := shaHex(actualBytes); digest != expectedDigest {
		return fmt.Errorf("%s digest got %s, want %s", name, digest, expectedDigest)
	}
	return nil
}

func checkBlob(descriptor any, raw []byte, name string) error {
	blob, ok := descriptor.(map[string]any)
	if !ok {
		return fmt.Errorf("%s descriptor is not an object", name)
	}
	sizeNumber, ok := blob["size_bytes"].(json.Number)
	if !ok {
		return fmt.Errorf("%s size is not an integer", name)
	}
	size, err := strconv.ParseInt(sizeNumber.String(), 10, 64)
	if err != nil || size != int64(len(raw)) {
		return fmt.Errorf("%s size does not match exact bytes", name)
	}
	if blob["sha256"] != shaHex(raw) {
		return fmt.Errorf("%s SHA-256 does not match exact bytes", name)
	}
	return nil
}

var canonicalSeconds = regexp.MustCompile(`^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$`)

func secondsToMilliseconds(seconds string) (int64, error) {
	if !canonicalSeconds.MatchString(seconds) {
		return 0, fmt.Errorf("non-canonical timeout %q", seconds)
	}
	value, ok := new(big.Rat).SetString(seconds)
	if !ok || value.Sign() <= 0 {
		return 0, fmt.Errorf("invalid positive decimal timeout %q", seconds)
	}
	scaled := new(big.Rat).Mul(value, big.NewRat(1000, 1))
	quotient, remainder := new(big.Int), new(big.Int)
	quotient.QuoRem(scaled.Num(), scaled.Denom(), remainder)
	if remainder.Sign() != 0 {
		quotient.Add(quotient, big.NewInt(1))
	}
	if !quotient.IsInt64() {
		return 0, fmt.Errorf("timeout %q overflows", seconds)
	}
	milliseconds := quotient.Int64()
	if milliseconds < 1 || milliseconds > 3600000 {
		return 0, fmt.Errorf("timeout %q is outside bounds", seconds)
	}
	return milliseconds, nil
}

func verify(golden *Golden) error {
	source, err := base64.StdEncoding.DecodeString(golden.SourceBase64)
	if err != nil {
		return fmt.Errorf("source base64: %w", err)
	}
	if golden.Request["candidate_sha256"] != shaHex(source) {
		return errors.New("candidate digest does not bind exact source bytes")
	}
	requestDoc, err := requestDocument(golden.Request)
	if err != nil {
		return err
	}
	if err := checkDocument(
		requestDoc, golden.RequestDigestDocument, golden.RequestCanonicalJSON,
		golden.RequestDigest, "request",
	); err != nil {
		return err
	}

	receiptDoc := receiptDocument(golden.Receipt.RequestID, golden.Receipt.AttemptID, golden.RequestDigest)
	if err := checkDocument(
		receiptDoc, golden.Receipt.Document, golden.Receipt.CanonicalJSON,
		golden.Receipt.ReceiptID, "receipt",
	); err != nil {
		return err
	}

	report, err := base64.StdEncoding.DecodeString(golden.Signature.ReportBase64)
	if err != nil {
		return fmt.Errorf("report base64: %w", err)
	}
	evidence, err := base64.StdEncoding.DecodeString(golden.Signature.EvidenceBase64)
	if err != nil {
		return fmt.Errorf("evidence base64: %w", err)
	}
	if err := checkBlob(golden.Signature.UnsignedResult["report"], report, "report"); err != nil {
		return err
	}
	if err := checkBlob(golden.Signature.UnsignedResult["evidence"], evidence, "evidence"); err != nil {
		return err
	}
	resultDoc, err := resultDocument(golden.Signature.UnsignedResult, golden.Signature.KeyID)
	if err != nil {
		return err
	}
	canonicalResult, err := canonical(resultDoc)
	if err != nil {
		return fmt.Errorf("result canonicalization: %w", err)
	}
	expectedResult, err := canonical(golden.Signature.Document)
	if err != nil || !bytes.Equal(canonicalResult, expectedResult) {
		return errors.New("result independently constructed document differs")
	}
	if string(canonicalResult) != golden.Signature.CanonicalJSON {
		return errors.New("result canonical JSON differs")
	}
	key, err := hex.DecodeString(golden.Signature.KeyHex)
	if err != nil || len(key) < 32 {
		return errors.New("invalid HMAC golden key")
	}
	mac := hmac.New(sha256.New, key)
	mac.Write(canonicalResult)
	actualSignature := hex.EncodeToString(mac.Sum(nil))
	if !hmac.Equal([]byte(actualSignature), []byte(golden.Signature.HMACSHA256)) {
		return fmt.Errorf("HMAC got %s, want %s", actualSignature, golden.Signature.HMACSHA256)
	}

	for _, vector := range golden.Timeouts {
		actual, err := secondsToMilliseconds(vector.Seconds)
		if err != nil || actual != vector.Milliseconds {
			return fmt.Errorf("timeout %s got %d (err=%v), want %d", vector.Seconds, actual, err, vector.Milliseconds)
		}
	}
	for _, seconds := range golden.RejectedTimeouts {
		if _, err := secondsToMilliseconds(seconds); err == nil {
			return fmt.Errorf("invalid timeout %q was accepted", seconds)
		}
	}
	return nil
}

func verifyBlob(golden *BlobGolden) error {
	requestDoc, err := blobRequestDocument(golden.Request)
	if err != nil {
		return err
	}
	if err := checkDocument(
		requestDoc, golden.RequestDigestDocument, golden.RequestCanonicalJSON,
		golden.RequestDigest, "blob request",
	); err != nil {
		return err
	}

	candidate := golden.Request["candidate_digest"]
	execution, _ := golden.Request["execution"].(map[string]any)
	group := execution["device_group"]
	receiptDoc := blobReceiptDocument(
		golden.Receipt.RequestID, golden.Receipt.AttemptID, golden.RequestDigest, candidate, group,
	)
	if err := checkDocument(
		receiptDoc, golden.Receipt.Document, golden.Receipt.CanonicalJSON,
		golden.Receipt.ReceiptID, "blob receipt",
	); err != nil {
		return err
	}

	report, err := base64.StdEncoding.DecodeString(golden.Signature.ReportBase64)
	if err != nil {
		return fmt.Errorf("blob report base64: %w", err)
	}
	evidence, err := base64.StdEncoding.DecodeString(golden.Signature.EvidenceBase64)
	if err != nil {
		return fmt.Errorf("blob evidence base64: %w", err)
	}
	if err := checkBlob(golden.Signature.UnsignedResult["report"], report, "blob report"); err != nil {
		return err
	}
	if err := checkBlob(golden.Signature.UnsignedResult["evidence"], evidence, "blob evidence"); err != nil {
		return err
	}
	resultDoc, err := blobResultDocument(golden.Signature.UnsignedResult, golden.Signature.KeyID)
	if err != nil {
		return err
	}
	canonicalResult, err := canonical(resultDoc)
	if err != nil {
		return fmt.Errorf("blob result canonicalization: %w", err)
	}
	expectedResult, err := canonical(golden.Signature.Document)
	if err != nil || !bytes.Equal(canonicalResult, expectedResult) {
		return errors.New("blob result independently constructed document differs")
	}
	if string(canonicalResult) != golden.Signature.CanonicalJSON {
		return errors.New("blob result canonical JSON differs")
	}
	key, err := hex.DecodeString(golden.Signature.KeyHex)
	if err != nil || len(key) < 32 {
		return errors.New("invalid blob HMAC golden key")
	}
	mac := hmac.New(sha256.New, key)
	mac.Write(canonicalResult)
	actualSignature := hex.EncodeToString(mac.Sum(nil))
	if !hmac.Equal([]byte(actualSignature), []byte(golden.Signature.HMACSHA256)) {
		return fmt.Errorf("blob HMAC got %s, want %s", actualSignature, golden.Signature.HMACSHA256)
	}
	return nil
}

var bundleManifestFields = []string{"path", "role", "media_type", "size_bytes", "sha256"}

// bundleManifest rebuilds the canonical manifest the artifact-set digest covers.
// It sorts entries by path and never looks at the content form: the manifest is
// the stable identity of the exact bytes, not of how they were transported.
// safeBundlePath refuses a manifest path that could escape the attempt
// directory (absolute, backslashes, control characters, empty/dot segments,
// surrounding whitespace) or that exceeds the path limits.
func safeBundlePath(path string) error {
	if path == "" {
		return errors.New("bundle entry path must be a non-empty string")
	}
	if len([]byte(path)) > maxBundlePathBytes {
		return errors.New("bundle entry path exceeds its UTF-8 byte limit")
	}
	if strings.HasPrefix(path, "/") {
		return errors.New("bundle entry path must be relative")
	}
	if strings.Contains(path, "\\") {
		return errors.New("bundle entry path must use POSIX separators")
	}
	if strings.HasSuffix(path, "/") {
		return errors.New("bundle entry path must name a file")
	}
	for _, r := range path {
		if r < ' ' || r == 0x7f {
			return errors.New("bundle entry path must not contain control characters")
		}
	}
	segments := strings.Split(path, "/")
	if len(segments) > maxBundlePathSegments {
		return errors.New("bundle entry path is too deep")
	}
	for _, segment := range segments {
		if segment == "" || segment == "." || segment == ".." {
			return fmt.Errorf("unsafe bundle path segment in %q", path)
		}
		if strings.TrimSpace(segment) != segment {
			return errors.New("bundle entry path segment must not have surrounding whitespace")
		}
	}
	return nil
}

func bundleManifest(schemaID string, entries []any) (map[string]any, error) {
	if schemaID != "tcenv.remote.npu-bundle.artifact-bundle.1" {
		return nil, fmt.Errorf("unsupported bundle schema %q", schemaID)
	}
	manifest := make([]any, 0, len(entries))
	seen := map[string]bool{}
	for _, raw := range entries {
		entry, ok := raw.(map[string]any)
		if !ok {
			return nil, errors.New("bundle entry is not an object")
		}
		canonicalEntry := make(map[string]any, len(bundleManifestFields))
		for _, name := range bundleManifestFields {
			value, present := entry[name]
			if !present {
				return nil, fmt.Errorf("bundle entry field %s is missing", name)
			}
			canonicalEntry[name] = value
		}
		path, ok := canonicalEntry["path"].(string)
		if !ok || path == "" {
			return nil, errors.New("bundle entry path is not a non-empty string")
		}
		if err := safeBundlePath(path); err != nil {
			return nil, err
		}
		if seen[strings.ToLower(path)] {
			return nil, fmt.Errorf("duplicate bundle path %q", path)
		}
		seen[strings.ToLower(path)] = true
		// The entry domain is part of the manifest identity: a role the contract
		// does not know, a media type that is not a bounded ASCII media type, a
		// digest that is not 64 lowercase hex characters or a size outside the
		// declared total are all different documents, not unusual values.
		role, ok := canonicalEntry["role"].(string)
		if !ok || !bundleRoles[role] {
			return nil, errors.New("bundle entry role is not a supported public role")
		}
		mediaType, ok := canonicalEntry["media_type"].(string)
		if !ok || !bundleMediaTypePattern.MatchString(mediaType) {
			return nil, errors.New("bundle entry media type is not a bounded ASCII media type")
		}
		size, err := buildNumber(canonicalEntry["size_bytes"], "bundle entry size")
		if err != nil || size < 0 || size > maxBundleTotalBytes {
			return nil, errors.New("bundle entry size is outside its integer bounds")
		}
		if digest, ok := canonicalEntry["sha256"].(string); !ok || !isHex64(digest) {
			return nil, errors.New(
				"bundle entry sha256 must be 64 lowercase hexadecimal characters",
			)
		}
		manifest = append(manifest, canonicalEntry)
	}
	if len(manifest) == 0 {
		return nil, errors.New("bundle has no entries")
	}
	if len(manifest) > maxBundleEntries {
		return nil, errors.New("a bundle exceeds the entry-count limit")
	}
	total := int64(0)
	for _, raw := range manifest {
		size, err := buildNumber(raw.(map[string]any)["size_bytes"], "bundle entry size")
		if err != nil {
			return nil, err
		}
		total += size
	}
	if total > maxBundleTotalBytes {
		return nil, errors.New("a bundle exceeds the declared total byte limit")
	}
	sort.Slice(manifest, func(i, j int) bool {
		return manifest[i].(map[string]any)["path"].(string) <
			manifest[j].(map[string]any)["path"].(string)
	})
	return map[string]any{
		"schema_id":      schemaID,
		"schema_version": 1,
		"entries":        manifest,
	}, nil
}

func bundleRequestDocument(request map[string]any) (map[string]any, error) {
	fields := []string{
		"kind", "run_id", "session_id", "task", "dtype", "size", "target_id",
		"task_fingerprint", "candidate_digest", "execution", "artifact", "mode", "seed", "reps",
		"queue_timeout_ms", "candidate_timeout_ms", "cases", "fast_bench",
		"require_contract_launch", "validate", "measure_serial", "cache_policy",
	}
	if len(request) != len(fields) {
		return nil, fmt.Errorf("bundle request has %d fields, want %d", len(request), len(fields))
	}
	canonicalRequest := make(map[string]any, len(fields))
	for _, name := range fields {
		value, present := request[name]
		if !present {
			return nil, fmt.Errorf("bundle request field %s is missing", name)
		}
		canonicalRequest[name] = value
	}
	artifact, ok := canonicalRequest["artifact"].(map[string]any)
	if !ok || len(artifact) != 1 {
		return nil, errors.New("artifact must carry exactly one branch")
	}
	bundle, ok := artifact["bundle"].(map[string]any)
	if !ok {
		return nil, errors.New("this vector must carry the bundle branch")
	}
	execution, ok := canonicalRequest["execution"].(map[string]any)
	if !ok {
		return nil, errors.New("execution is not an object")
	}
	if execution["task_family"] != "graph" || execution["artifact_kind"] != "bundle" ||
		execution["launch_kind"] != "GRAPH" || execution["launch_subtype"] != "SERVICE" {
		return nil, errors.New("execution shape is not the graph/service shape")
	}
	if execution["schema_id"] != "tcenv.remote.npu-bundle.execution.1" {
		return nil, errors.New("execution schema id is not the version 4 schema")
	}
	schemaID, ok := bundle["schema_id"].(string)
	if !ok {
		return nil, errors.New("bundle schema id is not a string")
	}
	entries, ok := bundle["entries"].([]any)
	if !ok {
		return nil, errors.New("bundle entries are not an array")
	}
	manifest, err := bundleManifest(schemaID, entries)
	if err != nil {
		return nil, err
	}
	manifestBytes, err := canonical(manifest)
	if err != nil {
		return nil, fmt.Errorf("bundle manifest canonicalization: %w", err)
	}
	artifactSet := shaHex(manifestBytes)
	if bundle["artifact_set_digest"] != artifactSet {
		return nil, errors.New("bundle artifact_set_digest does not bind the manifest")
	}
	candidate, ok := canonicalRequest["candidate_digest"].(map[string]any)
	if !ok {
		return nil, errors.New("candidate_digest is not an object")
	}
	if candidate["algorithm"] != "sha256-bundle-v1" || candidate["value"] != artifactSet {
		return nil, errors.New("candidate digest does not bind the artifact set")
	}
	return map[string]any{
		"document_type":    "tcenv.remote.npu-bundle.request-digest",
		"document_version": 4,
		"digest_algorithm": "sha256-rfc8785-v1",
		"request":          canonicalRequest,
	}, nil
}

func bundleReceiptDocument(requestID, attemptID, requestDigest string, candidate, group any) map[string]any {
	return map[string]any{
		"document_type":    "tcenv.remote.npu-bundle.receipt-id",
		"document_version": 4,
		"digest_algorithm": "sha256-rfc8785-v1",
		"request_id":       requestID,
		"attempt_id":       attemptID,
		"request_digest":   requestDigest,
		"candidate_digest": candidate,
		"device_group":     group,
	}
}

func bundleResultDocument(unsigned map[string]any, keyID string) map[string]any {
	return map[string]any{
		"document_type":       "tcenv.remote.npu-bundle.result-signature",
		"document_version":    4,
		"signature_algorithm": "hmac-sha256-v1",
		"key_id":              keyID,
		"provenance":          unsigned["provenance"],
		"report":              unsigned["report"],
		"evidence":            unsigned["evidence"],
		"receipt":             unsigned["receipt"],
	}
}

// verifyBundleEvidence independently checks the P1 completeness rules that make
// a device measurement publishable: exact coverage of the declared
// device x rank x die coordinates, an aggregate that is the slowest span, no
// invented rank/die dimension, and a device time that is the one the report
// claims -- never the host launch time.
func verifyBundleEvidence(evidence map[string]any, artifactSet string) error {
	if evidence["schema_id"] != "tcenv.remote.npu-bundle.timing-evidence.2" {
		return errors.New("evidence schema id is not the version 2 schema")
	}
	candidate, ok := evidence["candidate_digest"].(map[string]any)
	if !ok || candidate["value"] != artifactSet || candidate["algorithm"] != "sha256-bundle-v1" {
		return errors.New("evidence candidate digest does not bind the artifact set")
	}
	if evidence["artifact_set_digest"] != artifactSet {
		return errors.New("evidence artifact_set_digest differs from the artifact set")
	}
	if evidence["complete"] != true {
		return errors.New("evidence is not complete")
	}
	if evidence["publication_allowed"] != false {
		return errors.New("a simulated vector must not be publishable")
	}
	if err := verifyEvidenceCompleteness(evidence); err != nil {
		return err
	}
	timing, ok := evidence["timing"].(map[string]any)
	if !ok {
		return errors.New("evidence timing is missing")
	}
	deviceElapsed, ok := timing["device_elapsed_ns"].(json.Number)
	if !ok {
		return errors.New("device elapsed is not an integer")
	}
	device, err := strconv.ParseInt(deviceElapsed.String(), 10, 64)
	if err != nil {
		return err
	}
	spans := evidence["spans"].([]any)
	var aggregate int64
	for _, raw := range spans {
		span := raw.(map[string]any)
		elapsed, ok := span["elapsed_ns"].(json.Number)
		if !ok {
			return errors.New("span elapsed is not an integer")
		}
		value, err := strconv.ParseInt(elapsed.String(), 10, 64)
		if err != nil {
			return err
		}
		if value > aggregate {
			aggregate = value
		}
	}
	if device != aggregate {
		return errors.New("device_elapsed_ns is not the aggregate of the declared spans")
	}
	hostElapsed, ok := timing["host_api_elapsed_ns"].(json.Number)
	if !ok {
		return errors.New("host api elapsed is not an integer")
	}
	host, err := strconv.ParseInt(hostElapsed.String(), 10, 64)
	if err != nil {
		return err
	}
	if device <= 0 || host <= 0 {
		return errors.New("host and device elapsed must both be observed")
	}
	if timing["synchronization_completed"] != true || timing["source"] != "DEVICE_EVENT" {
		return errors.New("evidence is not a completed device-event observation")
	}
	if timing["clock_domain"] != spans[0].(map[string]any)["clock_domain"] {
		return errors.New("span and aggregate clock domains differ")
	}
	return nil
}

// verifyEvidenceCompleteness re-derives the declared topology coverage: every
// declared device, times every declared rank (when the topology has ranks),
// times every declared die, exactly once.  A shape without a rank or die
// dimension needs one span per device and must not report the dimension it
// does not have.
func verifyEvidenceCompleteness(evidence map[string]any) error {
	topology, ok := evidence["topology"].(map[string]any)
	if !ok {
		return errors.New("evidence topology is missing")
	}
	rawDevices, ok := topology["declared_devices"].([]any)
	if !ok || len(rawDevices) == 0 {
		return errors.New("evidence declares no device")
	}
	declared := make([]string, 0, len(rawDevices))
	declaredSeen := map[string]bool{}
	for _, raw := range rawDevices {
		name, ok := raw.(string)
		if !ok || name == "" {
			return errors.New("declared device id is not text")
		}
		if declaredSeen[name] {
			return errors.New("the declared topology lists the same device twice")
		}
		declaredSeen[name] = true
		declared = append(declared, name)
	}
	// ``logical_devices`` is part of the declared topology, so it has to agree
	// with the device list the spans are checked against.
	logicalValue, ok := topology["logical_devices"].(json.Number)
	if !ok {
		return errors.New("evidence topology has no logical device count")
	}
	logicalDevices, err := strconv.ParseInt(logicalValue.String(), 10, 64)
	if err != nil || logicalDevices < 1 {
		return errors.New("evidence topology logical device count is out of range")
	}
	if int(logicalDevices) != len(declared) {
		return errors.New("logical_devices disagrees with the declared device list")
	}
	dimension := func(name string) (int64, bool, error) {
		raw := topology[name]
		if raw == nil {
			return 0, false, nil
		}
		value, ok := raw.(json.Number)
		if !ok {
			return 0, false, fmt.Errorf("topology %s is not an integer", name)
		}
		parsed, err := strconv.ParseInt(value.String(), 10, 64)
		if err != nil || parsed < 1 {
			return 0, false, fmt.Errorf("topology %s is out of range", name)
		}
		return parsed, true, nil
	}
	ranks, hasRanks, err := dimension("required_rank_spans")
	if err != nil {
		return err
	}
	dies, hasDies, err := dimension("required_die_spans")
	if err != nil {
		return err
	}
	type coordinate struct {
		device  string
		rank    int64
		die     int64
		present bool
	}
	// -1 is only the *internal* marker for "this dimension does not exist for
	// this vendor shape"; it is never accepted as input.  A span that reports
	// ``rank: -1`` must be rejected exactly like the Python validator does
	// instead of being folded into "no rank dimension".
	coordinateValue := func(span map[string]any, name string) (int64, error) {
		raw, present := span[name]
		if !present {
			return 0, fmt.Errorf("span %s is missing", name)
		}
		if raw == nil {
			return -1, nil
		}
		value, ok := raw.(json.Number)
		if !ok {
			return 0, fmt.Errorf("span %s is not an integer", name)
		}
		parsed, err := strconv.ParseInt(value.String(), 10, 64)
		if err != nil || parsed < 0 {
			return 0, fmt.Errorf("span %s is outside its integer bounds", name)
		}
		return parsed, nil
	}
	rawSpans, ok := evidence["spans"].([]any)
	if !ok || len(rawSpans) == 0 {
		return errors.New("evidence is missing a declared device span")
	}
	seen := map[coordinate]bool{}
	for _, raw := range rawSpans {
		span, ok := raw.(map[string]any)
		if !ok {
			return errors.New("span is not an object")
		}
		device, ok := span["device_id"].(string)
		if !ok {
			return errors.New("span device id is not text")
		}
		rank, err := coordinateValue(span, "rank")
		if err != nil {
			return err
		}
		die, err := coordinateValue(span, "die")
		if err != nil {
			return err
		}
		if rank >= 0 && !hasRanks {
			return errors.New("a service vendor must not invent rank spans")
		}
		if die >= 0 && !hasDies {
			return errors.New("a service vendor must not invent die spans")
		}
		if hasRanks && rank >= ranks {
			return errors.New("a span reports a rank the declared topology does not cover")
		}
		if hasDies && die >= dies {
			return errors.New("a span reports a die the declared topology does not cover")
		}
		key := coordinate{device: device, rank: rank, die: die, present: true}
		if seen[key] {
			return errors.New("the same device/rank/die coordinate is reported twice")
		}
		seen[key] = true
	}
	expected := map[coordinate]bool{}
	rankValues := []int64{-1}
	if hasRanks {
		rankValues = make([]int64, 0, ranks)
		for index := int64(0); index < ranks; index++ {
			rankValues = append(rankValues, index)
		}
	}
	dieValues := []int64{-1}
	if hasDies {
		dieValues = make([]int64, 0, dies)
		for index := int64(0); index < dies; index++ {
			dieValues = append(dieValues, index)
		}
	}
	for _, device := range declared {
		for _, rank := range rankValues {
			for _, die := range dieValues {
				expected[coordinate{device: device, rank: rank, die: die, present: true}] = true
			}
		}
	}
	for key := range expected {
		if !seen[key] {
			if key.rank >= 0 {
				return fmt.Errorf("completed evidence is missing a declared rank span for %s/rank %d", key.device, key.rank)
			}
			return fmt.Errorf("completed evidence is missing a declared device span for %s", key.device)
		}
	}
	if len(seen) != len(expected) {
		return errors.New("evidence reports a span the declared topology does not cover")
	}
	return nil
}

// verifyCompletenessVectors re-derives every shared vector's verdict.
func verifyCompletenessVectors(vectors []EvidenceVector) error {
	if len(vectors) == 0 {
		return errors.New("the bundle golden carries no completeness vectors")
	}
	for _, vector := range vectors {
		err := verifyEvidenceCompleteness(vector.Document)
		if vector.Valid && err != nil {
			return fmt.Errorf("vector %q is valid but Go refuses it: %w", vector.Name, err)
		}
		if !vector.Valid && err == nil {
			return fmt.Errorf("vector %q is invalid but Go accepts it", vector.Name)
		}
	}
	return nil
}

func verifyBundle(golden *BundleGolden) error {
	requestDoc, err := bundleRequestDocument(golden.Request)
	if err != nil {
		return err
	}
	if err := checkDocument(
		requestDoc, golden.RequestDigestDocument, golden.RequestCanonicalJSON,
		golden.RequestDigest, "bundle request",
	); err != nil {
		return err
	}
	artifactSet := golden.Request["candidate_digest"].(map[string]any)["value"]
	if artifactSet != golden.ArtifactSetDigest {
		return errors.New("bundle artifact-set digest differs from the candidate digest")
	}
	manifest, err := bundleManifest(
		golden.BundleManifest["schema_id"].(string),
		golden.Request["artifact"].(map[string]any)["bundle"].(map[string]any)["entries"].([]any),
	)
	if err != nil {
		return err
	}
	manifestBytes, err := canonical(manifest)
	if err != nil {
		return err
	}
	if shaHex(manifestBytes) != golden.ArtifactSetDigest {
		return errors.New("artifact-set digest does not bind the canonical manifest")
	}
	expectedManifest, err := canonical(golden.BundleManifest)
	if err != nil {
		return err
	}
	if !bytes.Equal(manifestBytes, expectedManifest) {
		return errors.New("independently constructed bundle manifest differs")
	}
	evidenceDescriptor, ok := golden.Signature.UnsignedResult["evidence"].(map[string]any)
	if !ok {
		return errors.New("bundle result is missing the evidence descriptor")
	}
	// The descriptor names what the bytes are before anything parses them.  A
	// consistently re-signed vector with the version-1 schema/media type must be
	// refused here, not merely by an HMAC that happens to disagree.
	if evidenceDescriptor["schema_id"] != "tcenv.remote.npu-bundle.timing-evidence.2" {
		return errors.New("bundle evidence descriptor is not the version 2 schema identity")
	}
	if evidenceDescriptor["media_type"] != "application/vnd.tcenv.npu-bundle.timing-evidence.v2+json" {
		return errors.New("bundle evidence descriptor is not the version 2 media type")
	}
	// The report descriptor is the other signed identity.  Python's
	// ``result_signature_document`` refuses any report schema other than
	// ``tcenv.eval-report.v3``, so a consistently rewritten and re-signed vector
	// must fail here for the same reason instead of only by an HMAC mismatch.
	reportDescriptor, ok := golden.Signature.UnsignedResult["report"].(map[string]any)
	if !ok {
		return errors.New("bundle result is missing the report descriptor")
	}
	if reportDescriptor["schema_id"] != "tcenv.eval-report.v3" {
		return errors.New("bundle report descriptor is not the candidate report schema")
	}
	if err := verifyBundleEvidence(golden.Evidence, golden.ArtifactSetDigest); err != nil {
		return err
	}
	if err := verifyCompletenessVectors(golden.CompletenessVectors); err != nil {
		return err
	}
	if golden.Evidence["artifact_set_digest"] != golden.ArtifactSetDigest {
		return errors.New("evidence artifact-set digest differs")
	}
	receiptDoc := bundleReceiptDocument(
		golden.Receipt.RequestID, golden.Receipt.AttemptID, golden.RequestDigest,
		golden.Request["candidate_digest"], golden.Signature.UnsignedResult["receipt"].(map[string]any)["device_group"],
	)
	if err := checkDocument(
		receiptDoc, golden.Receipt.Document, golden.Receipt.CanonicalJSON,
		golden.Receipt.ReceiptID, "bundle receipt",
	); err != nil {
		return err
	}
	report, err := base64.StdEncoding.DecodeString(golden.Signature.ReportBase64)
	if err != nil {
		return err
	}
	evidence, err := base64.StdEncoding.DecodeString(golden.Signature.EvidenceBase64)
	if err != nil {
		return err
	}
	if err := checkBlob(golden.Signature.UnsignedResult["report"], report, "bundle report"); err != nil {
		return err
	}
	if err := checkBlob(golden.Signature.UnsignedResult["evidence"], evidence, "bundle evidence"); err != nil {
		return err
	}
	var reportDocument map[string]any
	reportDecoder := json.NewDecoder(bytes.NewReader(report))
	reportDecoder.UseNumber()
	if err := reportDecoder.Decode(&reportDocument); err != nil {
		return fmt.Errorf("bundle report is not JSON: %w", err)
	}
	// The signed bytes are what a consumer actually reads.  Decode them, run the
	// full semantic checks on *that* document, and require it to be the same
	// document the golden keeps separately -- otherwise a tampered payload with a
	// matching descriptor, canonical document and HMAC would be verified against
	// an untouched detached copy.
	var signedEvidence map[string]any
	evidenceDecoder := json.NewDecoder(bytes.NewReader(evidence))
	evidenceDecoder.UseNumber()
	if err := evidenceDecoder.Decode(&signedEvidence); err != nil {
		return fmt.Errorf("bundle evidence payload is not JSON: %w", err)
	}
	signedCanonical, err := canonical(signedEvidence)
	if err != nil {
		return err
	}
	detachedCanonical, err := canonical(golden.Evidence)
	if err != nil {
		return err
	}
	if !bytes.Equal(signedCanonical, detachedCanonical) {
		return errors.New("signed evidence payload differs from the documented evidence")
	}
	if err := verifyBundleEvidence(signedEvidence, golden.ArtifactSetDigest); err != nil {
		return err
	}
	timing := signedEvidence["timing"].(map[string]any)
	deviceElapsed := timing["device_elapsed_ns"].(json.Number)
	candidateUs, ok := reportDocument["candidate_us"].(json.Number)
	if !ok {
		return errors.New("bundle report has no candidate_us")
	}
	nanos, err := strconv.ParseInt(deviceElapsed.String(), 10, 64)
	if err != nil {
		return err
	}
	micros, err := strconv.ParseFloat(candidateUs.String(), 64)
	if err != nil {
		return err
	}
	if micros != float64(nanos)/1000.0 {
		return errors.New("report candidate_us does not come from the device elapsed time")
	}
	resultDoc := bundleResultDocument(golden.Signature.UnsignedResult, golden.Signature.KeyID)
	expectedResult, err := canonical(golden.Signature.Document)
	if err != nil {
		return err
	}
	canonicalResult, err := canonical(resultDoc)
	if err != nil {
		return err
	}
	if string(canonicalResult) != golden.Signature.CanonicalJSON {
		return errors.New("bundle result canonical JSON differs")
	}
	if !bytes.Equal(canonicalResult, expectedResult) {
		return errors.New("independently constructed bundle result differs")
	}
	key, err := hex.DecodeString(golden.Signature.KeyHex)
	if err != nil {
		return errors.New("invalid bundle HMAC golden key")
	}
	mac := hmac.New(sha256.New, key)
	mac.Write(canonicalResult)
	actual := hex.EncodeToString(mac.Sum(nil))
	if !hmac.Equal([]byte(actual), []byte(golden.Signature.HMACSHA256)) {
		return fmt.Errorf("bundle HMAC got %s, want %s", actual, golden.Signature.HMACSHA256)
	}
	return nil
}

func main() {
	goldenPath := flag.String("golden", "", "path to evaluator_v1.json")
	flag.Parse()
	if *goldenPath == "" {
		fmt.Fprintln(os.Stderr, "-golden is required")
		os.Exit(2)
	}
	file, err := os.Open(*goldenPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	decoder.UseNumber()
	var golden Golden
	if err := decoder.Decode(&golden); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		fmt.Fprintln(os.Stderr, "golden has trailing JSON")
		os.Exit(1)
	}
	if golden.Build != nil {
		if err := verifyBuild(golden.Build); err != nil {
			fmt.Fprintln(os.Stderr, "Go build contract verification failed:", err)
			os.Exit(1)
		}
		fmt.Printf("ok build artifact_set=%s request=%s receipt=%s producer=%s\n",
			golden.Build.ArtifactSetDigest, golden.Build.RequestDigest,
			golden.Build.Receipt.ReceiptID, golden.Build.ProducerRecord["evidence_sha256"])
		return
	}
	if golden.Bundle != nil {
		if err := verifyBundle(golden.Bundle); err != nil {
			fmt.Fprintln(os.Stderr, "Go bundle contract verification failed:", err)
			os.Exit(1)
		}
		fmt.Printf("ok bundle artifact_set=%s request=%s receipt=%s signature=%s\n",
			golden.Bundle.ArtifactSetDigest, golden.Bundle.RequestDigest,
			golden.Bundle.Receipt.ReceiptID, golden.Bundle.Signature.HMACSHA256)
		return
	}
	if golden.Blob != nil {
		if err := verifyBlob(golden.Blob); err != nil {
			fmt.Fprintln(os.Stderr, "Go blob contract verification failed:", err)
			os.Exit(1)
		}
		fmt.Printf("ok blob request=%s receipt=%s signature=%s\n",
			golden.Blob.RequestDigest, golden.Blob.Receipt.ReceiptID, golden.Blob.Signature.HMACSHA256)
		return
	}
	if err := verify(&golden); err != nil {
		fmt.Fprintln(os.Stderr, "Go contract verification failed:", err)
		os.Exit(1)
	}
	fmt.Printf("ok request=%s receipt=%s signature=%s\n",
		golden.RequestDigest, golden.Receipt.ReceiptID, golden.Signature.HMACSHA256)
}

// -- P1c build contract (document version 5) ------------------------------

// BuildGolden freezes the P1c build-contract documents.  The Go side rebuilds
// the request digest document (including the independent `build` field), the
// producer evidence link and the received build-evidence rules from the raw
// fields, so the Python validator is not the only implementation of this
// contract.
type BuildGolden struct {
	Request               map[string]any `json:"request"`
	ProducerEvidence      map[string]any `json:"producer_evidence"`
	ProducerRecord        map[string]any `json:"producer_record"`
	BundleManifest        map[string]any `json:"bundle_manifest"`
	ArtifactSetDigest     string         `json:"artifact_set_digest"`
	RequestDigestDocument map[string]any `json:"request_digest_document"`
	RequestCanonicalJSON  string         `json:"request_canonical_json"`
	RequestDigest         string         `json:"request_digest"`
	Receipt               ReceiptGolden  `json:"receipt"`
	Evidence              map[string]any `json:"evidence"`
}

var buildRequestFields = []string{
	"kind", "run_id", "session_id", "task", "dtype", "size", "target_id",
	"task_fingerprint", "candidate_digest", "execution", "artifact", "build",
	"mode", "seed", "reps", "queue_timeout_ms", "candidate_timeout_ms", "cases",
	"fast_bench", "require_contract_launch", "validate", "measure_serial", "cache_policy",
}

var buildDocumentFields = []string{"schema_id", "mode", "provenance", "producer"}

var buildProvenanceFields = []string{
	"schema_id", "builder_id", "builder_version", "build_host", "build_profile",
	"toolchain", "compile_profile", "source_sha256", "artifact_set_digest",
}

var buildProducerRecordFields = []string{
	"schema_id", "status", "mode", "duration_ms", "log", "evidence_sha256",
}

// buildProvenanceDocument rebuilds the provenance identity field by field.  The
// closed field set is the point: a provenance that grew a field the request
// digest does not cover would be an unreviewed identity.
func buildProvenanceDocument(
	value any, mode string, artifactSet string, sourceSHA string,
) (map[string]any, error) {
	provenance, ok := value.(map[string]any)
	if !ok {
		return nil, errors.New("build provenance is not an object")
	}
	if len(provenance) != len(buildProvenanceFields) {
		return nil, fmt.Errorf(
			"build provenance has %d fields, want %d", len(provenance), len(buildProvenanceFields),
		)
	}
	document := make(map[string]any, len(buildProvenanceFields))
	for _, name := range buildProvenanceFields {
		field, present := provenance[name]
		if !present {
			return nil, fmt.Errorf("build provenance field %s is missing", name)
		}
		document[name] = field
	}
	if document["schema_id"] != "tcenv.remote.npu-build.provenance.1" {
		return nil, errors.New("build provenance schema id is not the version 1 identity")
	}
	if document["build_host"] != "control_host" && document["build_host"] != "evaluator" {
		return nil, errors.New("build provenance build_host is not a known placement")
	}
	// ``mode`` and ``build_host`` are one identity: a prebuilt artifact set was
	// compiled on the control host, an evaluator build inside the attempt.  The
	// pairing is not negotiable, so it is checked with the provenance.
	if mode == "prebuilt" && document["build_host"] != "control_host" {
		return nil, errors.New("build mode prebuilt requires build_host control_host")
	}
	if mode == "evaluator" && document["build_host"] != "evaluator" {
		return nil, errors.New("build mode evaluator requires build_host evaluator")
	}
	toolchain, ok := document["toolchain"].(map[string]any)
	if !ok || len(toolchain) != 3 {
		return nil, errors.New("build provenance toolchain is not {id,version,sdk_version}")
	}
	for _, name := range []string{"id", "version", "sdk_version"} {
		if _, present := toolchain[name]; !present {
			return nil, fmt.Errorf("build provenance toolchain field %s is missing", name)
		}
	}
	// The identity fields are bounded, not free text: an id is an opaque
	// identifier, a version is bounded text, and the compile profile is a
	// bounded scalar object.  Without this the provenance would accept values
	// the production contract refuses (a number as a builder id, a 65-byte
	// version, a list as a compile profile, ...).
	if _, err := buildIdentifier(document["builder_id"], "build.provenance.builder_id"); err != nil {
		return nil, err
	}
	if _, err := buildText(document["builder_version"], "build.provenance.builder_version", 64); err != nil {
		return nil, err
	}
	if buildHost, err := buildText(document["build_host"], "build.provenance.build_host", 32); err != nil {
		return nil, err
	} else if buildHost != "control_host" && buildHost != "evaluator" {
		return nil, errors.New("build.provenance.build_host must be control_host or evaluator")
	}
	if _, err := buildIdentifier(document["build_profile"], "build.provenance.build_profile"); err != nil {
		return nil, err
	}
	if _, err := buildIdentifier(toolchain["id"], "build.toolchain.id"); err != nil {
		return nil, err
	}
	if _, err := buildText(toolchain["version"], "build.toolchain.version", 64); err != nil {
		return nil, err
	}
	if _, err := buildText(toolchain["sdk_version"], "build.toolchain.sdk_version", 64); err != nil {
		return nil, err
	}
	if _, err := buildCompileProfile(document["compile_profile"]); err != nil {
		return nil, err
	}
	if digest, ok := document["source_sha256"].(string); !ok || !isHex64(digest) {
		return nil, errors.New("build.provenance.source_sha256 must be 64 lowercase hexadecimal characters")
	}
	if digest := document["artifact_set_digest"]; digest != nil {
		text, ok := digest.(string)
		if !ok || !isHex64(text) {
			return nil, errors.New(
				"build.provenance.artifact_set_digest must be 64 lowercase hexadecimal characters",
			)
		}
	}
	if sourceSHA != "" && document["source_sha256"] != sourceSHA {
		return nil, errors.New("build provenance source_sha256 does not bind the SOURCE entry")
	}
	if artifactSet != "" && document["artifact_set_digest"] != artifactSet {
		return nil, errors.New("build provenance artifact_set_digest does not bind the artifact set")
	}
	return document, nil
}

// buildRequestDocument rebuilds the version-5 request digest document.  It is
// the version-4 bundle document plus exactly one independent field, `build`;
// every other field set and the digest algorithm are unchanged.
func buildRequestDocument(request map[string]any) (map[string]any, error) {
	if len(request) != len(buildRequestFields) {
		return nil, fmt.Errorf(
			"build request has %d fields, want %d", len(request), len(buildRequestFields),
		)
	}
	canonicalRequest := make(map[string]any, len(buildRequestFields))
	for _, name := range buildRequestFields {
		value, present := request[name]
		if !present {
			return nil, fmt.Errorf("build request field %s is missing", name)
		}
		canonicalRequest[name] = value
	}
	artifact, ok := canonicalRequest["artifact"].(map[string]any)
	if !ok || len(artifact) != 1 {
		return nil, errors.New("artifact must carry exactly one branch")
	}
	bundle, ok := artifact["bundle"].(map[string]any)
	if !ok {
		return nil, errors.New("this vector must carry the bundle branch")
	}
	execution, ok := canonicalRequest["execution"].(map[string]any)
	if !ok {
		return nil, errors.New("execution is not an object")
	}
	if execution["schema_id"] != "tcenv.remote.npu-bundle.execution.1" ||
		execution["task_family"] != "kernel_module" ||
		execution["artifact_kind"] != "bundle" ||
		execution["launch_kind"] != "KERNEL" ||
		execution["launch_subtype"] != "SIMT" ||
		execution["synchronization"] != "stream-event" {
		return nil, errors.New("execution shape is not the prebuilt kernel_module shape")
	}
	// The task family derives the source media type; it is not negotiated.  A
	// kernel_module payload is text (the module itself is an artifact entry).
	if execution["source_media_type"] != buildModuleMediaType {
		return nil, fmt.Errorf(
			"execution.task_family 'kernel_module' requires source_media_type %q",
			buildModuleMediaType,
		)
	}
	schemaID, ok := bundle["schema_id"].(string)
	if !ok {
		return nil, errors.New("bundle schema id is not a string")
	}
	entries, ok := bundle["entries"].([]any)
	if !ok {
		return nil, errors.New("bundle entries are not an array")
	}
	manifest, err := bundleManifest(schemaID, entries)
	if err != nil {
		return nil, err
	}
	manifestBytes, err := canonical(manifest)
	if err != nil {
		return nil, fmt.Errorf("bundle manifest canonicalization: %w", err)
	}
	artifactSet := shaHex(manifestBytes)
	if bundle["artifact_set_digest"] != artifactSet {
		return nil, errors.New("bundle artifact_set_digest does not bind the manifest")
	}
	candidate, ok := canonicalRequest["candidate_digest"].(map[string]any)
	if !ok {
		return nil, errors.New("candidate_digest is not an object")
	}
	if candidate["algorithm"] != "sha256-bundle-v1" || candidate["value"] != artifactSet {
		return nil, errors.New("candidate digest does not bind the artifact set")
	}
	// A prebuilt artifact set describes exactly one candidate: one SOURCE entry
	// and at least one executable module.  Two SOURCE entries would leave the
	// candidate identity ambiguous (the provenance can only bind one digest).
	sourceSHA := ""
	sourceCount := 0
	moduleCount := 0
	for _, raw := range entries {
		entry, ok := raw.(map[string]any)
		if !ok {
			return nil, errors.New("bundle entry is not an object")
		}
		switch entry["role"] {
		case "SOURCE":
			sourceCount++
			sha, ok := entry["sha256"].(string)
			if !ok {
				return nil, errors.New("SOURCE entry sha256 is not a string")
			}
			if sourceCount == 1 {
				sourceSHA = sha
			}
		case "EXECUTABLE_MODULE":
			moduleCount++
		}
	}
	if sourceCount != 1 {
		return nil, errors.New("a prebuilt artifact set requires exactly one SOURCE entry")
	}
	if moduleCount == 0 {
		return nil, errors.New("a prebuilt artifact set requires an EXECUTABLE_MODULE entry")
	}
	build, ok := canonicalRequest["build"].(map[string]any)
	if !ok {
		return nil, errors.New("build is not an object")
	}
	if len(build) != len(buildDocumentFields) {
		return nil, fmt.Errorf(
			"build document has %d fields, want %d", len(build), len(buildDocumentFields),
		)
	}
	for _, name := range buildDocumentFields {
		if _, present := build[name]; !present {
			return nil, fmt.Errorf("build field %s is missing", name)
		}
	}
	if build["schema_id"] != "tcenv.remote.npu-build.build-request.1" {
		return nil, errors.New("build schema id is not the version 1 identity")
	}
	buildMode, modeIsString := build["mode"].(string)
	if !modeIsString || buildMode != "prebuilt" {
		// This vector carries a prebuilt artifact set and its producer link, so
		// an evaluator mode is not another legal encoding of the same request:
		// an evaluator request has no artifact set and no producer at all.
		return nil, errors.New(
			"this vector requires the prebuilt build shape (mode prebuilt and a producer link)",
		)
	}
	if build["producer"] == nil {
		return nil, errors.New("a prebuilt build request requires the producer evidence link")
	}
	if _, err := buildProvenanceDocument(
		build["provenance"], buildMode, artifactSet, sourceSHA,
	); err != nil {
		return nil, err
	}
	if raw, present := build["producer"]; present && raw != nil {
		record, ok := raw.(map[string]any)
		if !ok || len(record) != len(buildProducerRecordFields) {
			return nil, errors.New("producer link is not the closed record shape")
		}
		for _, name := range buildProducerRecordFields {
			if _, present := record[name]; !present {
				return nil, fmt.Errorf("producer link field %s is missing", name)
			}
		}
		if record["schema_id"] != "tcenv.remote.npu-build.producer-evidence-link.1" {
			return nil, errors.New("producer link schema id is not the version 1 identity")
		}
		if record["status"] != "ok" || record["mode"] != buildMode {
			return nil, errors.New(
				"a prebuilt producer link must describe a successful control-host compile",
			)
		}
		if _, err := boundedBuildNumber(
			record["duration_ms"], "build.producer.duration_ms", 0, maxBuildDurationMs,
		); err != nil {
			return nil, err
		}
		// The producer's log is optional (a builder may emit no diagnostics) but
		// a present one is the same bounded shape the receiver has to produce.
		if rawLog := record["log"]; rawLog != nil {
			producerLog, ok := rawLog.(map[string]any)
			if !ok {
				return nil, errors.New("build.producer.log is not an object")
			}
			if err := closedFields(producerLog, buildLogFields, nil, "build.producer.log"); err != nil {
				return nil, err
			}
			if digest, ok := producerLog["sha256"].(string); !ok || !isHex64(digest) {
				return nil, errors.New(
					"build.producer.log sha256 is not a lowercase 64-character digest",
				)
			}
			if _, err := boundedBuildNumber(
				producerLog["size_bytes"], "build.producer.log.size_bytes", 0, maxBuildSourceBytes,
			); err != nil {
				return nil, err
			}
		}
	}
	return map[string]any{
		"document_type":    "tcenv.remote.npu-build.request-digest",
		"document_version": 5,
		"digest_algorithm": "sha256-rfc8785-v1",
		"request":          canonicalRequest,
	}, nil
}

// buildProducerDigest recomputes the producer link's evidence digest from the
// producer's own evidence document.  This is the cross-language half of the
// closure the Python side emits: the link is only a link if the document it
// names hashes to the value it carries.
func buildProducerDigest(document map[string]any) (string, error) {
	encoded, err := canonical(document)
	if err != nil {
		return "", err
	}
	return shaHex(encoded), nil
}

// buildIdentifier is the bounded opaque identifier the build contract allows for
// builder/profile/toolchain ids: 1..160 characters from [A-Za-z0-9._:-].
var buildIdentifierPattern = regexp.MustCompile(`^[A-Za-z0-9._:-]{1,160}$`)

func buildIdentifier(value any, name string) (string, error) {
	text, ok := value.(string)
	if !ok || !buildIdentifierPattern.MatchString(text) {
		return "", fmt.Errorf("%s must be a bounded opaque identifier", name)
	}
	return text, nil
}

func buildText(value any, name string, maxBytes int) (string, error) {
	text, ok := value.(string)
	if !ok || text == "" {
		return "", fmt.Errorf("%s must be a non-empty string", name)
	}
	if len([]byte(text)) > maxBytes {
		return "", fmt.Errorf("%s exceeds its UTF-8 byte limit", name)
	}
	return text, nil
}

// buildCompileProfile mirrors the bounded scalar object the compile profile is:
// at most 16 fields, ASCII keys of at most 64 bytes, and scalar values (string,
// boolean, safe integer or null).
func buildCompileProfile(value any) (map[string]any, error) {
	profile, ok := value.(map[string]any)
	if !ok || len(profile) > 16 {
		return nil, errors.New("build.provenance.compile_profile must be a bounded object")
	}
	for key, item := range profile {
		if key == "" || len(key) > 64 || !isASCII(key) {
			return nil, errors.New("timing diagnostic key is invalid")
		}
		switch typed := item.(type) {
		case string:
			if len([]byte(typed)) > 512 {
				return nil, errors.New("timing diagnostic exceeds its UTF-8 byte limit")
			}
		case bool, nil:
		case json.Number:
			parsed, err := strconv.ParseInt(typed.String(), 10, 64)
			if err != nil || parsed > maxBuildSafeInteger || parsed < -maxBuildSafeInteger {
				return nil, errors.New("timing diagnostic integer is unsafe")
			}
		default:
			return nil, errors.New("timing diagnostics must be scalar")
		}
	}
	return profile, nil
}

// isHex64 reports whether value is 64 lowercase hexadecimal characters.
func isHex64(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, char := range value {
		if !((char >= '0' && char <= '9') || (char >= 'a' && char <= 'f')) {
			return false
		}
	}
	return true
}

// sameDocument compares two raw JSON values by their canonical encodings.
func sameDocument(left, right any) (bool, error) {
	leftBytes, err := canonical(left)
	if err != nil {
		return false, err
	}
	rightBytes, err := canonical(right)
	if err != nil {
		return false, err
	}
	return bytes.Equal(leftBytes, rightBytes), nil
}

var buildEvidenceFields = []string{
	"schema_id", "mode", "status", "error_class", "provenance", "artifacts",
	"duration_ms", "log", "loaded",
}

var buildArtifactsFields = []string{"schema_id", "entries"}

var buildLogFields = []string{"sha256", "size_bytes"}

var buildLoadedFields = []string{"path", "sha256"}

// The build contract's integer domains, mirrored from the production validator:
// a duration is a bounded wall-clock value (0 means "this never ran") and a log
// size is a bounded byte count.
const (
	maxBuildDurationMs    = 3_600_000
	maxBuildSourceBytes   = 1_000_000
	maxBuildSafeInteger   = 9_007_199_254_740_991
	buildModuleMediaType  = "text/plain; charset=utf-8"
	maxBundleEntries      = 64
	maxBundlePathBytes    = 240
	maxBundlePathSegments = 16
	maxBundleTotalBytes   = 16_000_000
)

var bundleRoles = map[string]bool{
	"SOURCE": true, "EXECUTABLE_MODULE": true, "GRAPH_DEFINITION": true,
	"MODEL_DEFINITION": true, "PARAMETER_TABLE": true, "AUXILIARY": true,
}

// bundleMediaTypePattern is the bounded ASCII media type a bundle entry may
// declare (same character class as the production validator).
var bundleMediaTypePattern = regexp.MustCompile(`^[A-Za-z0-9!#$&^_.+*/;= -]{1,160}$`)

// closedFields refuses a document whose field set is not exactly the versioned
// one.  A build evidence that grew a field is an unreviewed identity even when
// every field it does have is valid, which is why the check runs before any
// value is compared.
func closedFields(value map[string]any, required, optional []string, name string) error {
	allowed := make(map[string]bool, len(required)+len(optional))
	for _, field := range required {
		allowed[field] = true
		if _, present := value[field]; !present {
			return fmt.Errorf("%s fields are missing or unknown", name)
		}
	}
	for _, field := range optional {
		allowed[field] = true
	}
	for field := range value {
		if !allowed[field] {
			return fmt.Errorf("%s fields are missing or unknown", name)
		}
	}
	return nil
}

// boundedBuildNumber parses a build-contract integer and applies its domain.
// Parsing alone would accept a negative log size or a duration the contract
// does not allow, which is exactly the gap a value-only check leaves open.
func boundedBuildNumber(value any, name string, low, high int64) (int64, error) {
	parsed, err := buildNumber(value, name)
	if err != nil {
		return 0, err
	}
	if parsed < low || parsed > high {
		return 0, fmt.Errorf("%s is outside its integer bounds", name)
	}
	return parsed, nil
}

func buildNumber(value any, name string) (int64, error) {
	number, ok := value.(json.Number)
	if !ok {
		return 0, fmt.Errorf("%s is not an integer", name)
	}
	parsed, err := strconv.ParseInt(number.String(), 10, 64)
	if err != nil {
		return 0, fmt.Errorf("%s is not an integer: %w", name, err)
	}
	return parsed, nil
}

// verifyBuildEvidence re-derives the received (prebuilt) build-evidence rules:
// the artifact set is reused, it repeats the accepted producer link, the link
// hashes its own producer evidence, the manifest and load record cover the
// executable module, and the receiver recorded its own verify/load cost.
func verifyBuildEvidence(
	evidence, requestBuild, requestArtifact, producerEvidence map[string]any, artifactSet string,
) error {
	build, ok := evidence["build"].(map[string]any)
	if !ok {
		return errors.New("evidence build document is missing")
	}
	if err := closedFields(
		build, buildEvidenceFields, []string{"producer"}, "build evidence",
	); err != nil {
		return err
	}
	if build["schema_id"] != "tcenv.remote.npu-build.evidence.1" {
		return errors.New("build evidence schema id is not the version 1 identity")
	}
	requestMode, ok := requestBuild["mode"].(string)
	if !ok || requestMode != "prebuilt" {
		return errors.New("this vector carries the prebuilt timing")
	}
	if build["mode"] != requestMode {
		return errors.New("build evidence mode differs from the accepted request")
	}
	if build["status"] != "reused" {
		return errors.New("a received prebuilt artifact set must be reported as reused")
	}
	if build["error_class"] != nil {
		return errors.New("a reused artifact set has no build error")
	}
	provenance, err := buildProvenanceDocument(
		build["provenance"], requestMode, artifactSet, "",
	)
	if err != nil {
		return err
	}
	expectedProvenance, err := buildProvenanceDocument(
		requestBuild["provenance"], requestMode, artifactSet, "",
	)
	if err != nil {
		return err
	}
	for _, name := range buildProvenanceFields {
		if name == "artifact_set_digest" {
			continue
		}
		same, err := sameDocument(provenance[name], expectedProvenance[name])
		if err != nil {
			return err
		}
		if !same {
			return fmt.Errorf("build evidence %s differs from the accepted request", name)
		}
	}
	producer, ok := build["producer"].(map[string]any)
	if !ok {
		return errors.New("a reused artifact set requires the producer evidence link")
	}
	requestProducer, ok := requestBuild["producer"].(map[string]any)
	if !ok {
		return errors.New("the accepted request has no producer link")
	}
	same, err := sameDocument(producer, requestProducer)
	if err != nil {
		return err
	}
	if !same {
		return errors.New("the received build evidence does not repeat the accepted producer link")
	}
	digest, err := buildProducerDigest(producerEvidence)
	if err != nil {
		return err
	}
	if producer["evidence_sha256"] != digest {
		return errors.New("the producer link does not bind its own evidence document")
	}
	if producer["duration_ms"] != producerEvidence["duration_ms"] {
		return errors.New("the producer link duration differs from the producer evidence")
	}
	sameLog, err := sameDocument(producer["log"], producerEvidence["log"])
	if err != nil {
		return err
	}
	if !sameLog {
		return errors.New("the producer link log differs from the producer evidence")
	}
	artifacts, ok := build["artifacts"].(map[string]any)
	if !ok {
		return errors.New("a reused artifact set requires its manifest identity")
	}
	if err := closedFields(
		artifacts, buildArtifactsFields, []string{"artifact_set_digest"}, "build.artifacts",
	); err != nil {
		return err
	}
	entries, ok := artifacts["entries"].([]any)
	if !ok || len(entries) == 0 {
		return errors.New("a reused artifact set requires its manifest entries")
	}
	// The evidence manifest is rebuilt from its own entries, so the digest it
	// declares is a claim this verifier can check instead of a value it trusts.
	artifactsSchema, ok := artifacts["schema_id"].(string)
	if !ok {
		return errors.New("build.artifacts schema_id is not a string")
	}
	manifest, err := bundleManifest(artifactsSchema, entries)
	if err != nil {
		return err
	}
	manifestBytes, err := canonical(manifest)
	if err != nil {
		return err
	}
	computed := shaHex(manifestBytes)
	// The declared digest is a claim about the entries that are actually here.
	if declared, present := artifacts["artifact_set_digest"]; present && declared != nil {
		declaredDigest, ok := declared.(string)
		if !ok {
			return errors.New("build.artifacts artifact_set_digest is not a string")
		}
		if declaredDigest != computed {
			return errors.New("build.artifacts artifact_set_digest does not bind its entries")
		}
	}
	// ...and the manifest has to be the artifact set the request was accepted
	// for, entry by entry: a receiver may not describe other bytes.
	acceptedBundle, ok := requestArtifact["bundle"].(map[string]any)
	if !ok {
		return errors.New("the accepted request has no prebuilt artifact set")
	}
	acceptedEntries, ok := acceptedBundle["entries"].([]any)
	if !ok {
		return errors.New("the accepted prebuilt bundle has no entries")
	}
	acceptedSchema, ok := acceptedBundle["schema_id"].(string)
	if !ok {
		return errors.New("the accepted prebuilt bundle schema id is not a string")
	}
	acceptedManifest, err := bundleManifest(acceptedSchema, acceptedEntries)
	if err != nil {
		return err
	}
	acceptedBytes, err := canonical(acceptedManifest)
	if err != nil {
		return err
	}
	if !bytes.Equal(manifestBytes, acceptedBytes) {
		return errors.New("build evidence entries differ from the accepted prebuilt bundle")
	}
	if computed != artifactSet {
		return errors.New("build evidence artifacts do not add up to the artifact set")
	}
	modules := map[string]string{}
	for _, raw := range entries {
		entry, ok := raw.(map[string]any)
		if !ok {
			return errors.New("build evidence entry is not an object")
		}
		if entry["role"] == "EXECUTABLE_MODULE" {
			sha, ok := entry["sha256"].(string)
			if !ok {
				return errors.New("executable module sha256 is not a string")
			}
			path, ok := entry["path"].(string)
			if !ok {
				return errors.New("executable module path is not a string")
			}
			modules[path] = sha
		}
	}
	if len(modules) == 0 {
		return errors.New("a reused artifact set requires an executable module")
	}
	known := map[string]string{}
	for _, raw := range entries {
		entry, ok := raw.(map[string]any)
		if !ok {
			return errors.New("build evidence entry is not an object")
		}
		path, _ := entry["path"].(string)
		sha, _ := entry["sha256"].(string)
		known[path] = sha
	}
	loaded, ok := build["loaded"].([]any)
	if !ok {
		return errors.New("build.loaded must be an array")
	}
	// The load record is a set of artifacts this attempt read back: each path
	// once, each digest the built one, and every executable module covered.  A
	// count check plus membership alone accepts a record that lists one module
	// twice and silently omits another.
	loadedPaths := map[string]bool{}
	for _, raw := range loaded {
		record, ok := raw.(map[string]any)
		if !ok {
			return errors.New("load record is not an object")
		}
		if err := closedFields(record, buildLoadedFields, nil, "build.loaded"); err != nil {
			return err
		}
		path, ok := record["path"].(string)
		if !ok {
			return errors.New("build.loaded path is not a string")
		}
		if err := safeBundlePath(path); err != nil {
			return err
		}
		sha, ok := record["sha256"].(string)
		if !ok || !isHex64(sha) {
			return errors.New("build.loaded.sha256 must be 64 lowercase hexadecimal characters")
		}
		if loadedPaths[path] {
			return errors.New("build.loaded reports the same artifact twice")
		}
		loadedPaths[path] = true
		knownSHA, produced := known[path]
		if !produced {
			return fmt.Errorf(
				"build.loaded names an artifact the build did not produce: %s", path,
			)
		}
		if knownSHA != sha {
			return errors.New("load record does not name the built module")
		}
	}
	for path := range modules {
		if !loadedPaths[path] {
			return errors.New("the receiver's load record does not cover the executable modules")
		}
	}
	// The receiver records its own verify/load cost; reusing the producer's
	// duration or log identity would make the two rows indistinguishable.
	receiverDuration, err := boundedBuildNumber(
		build["duration_ms"], "build.duration_ms", 0, maxBuildDurationMs,
	)
	if err != nil {
		return err
	}
	if receiverDuration <= 0 {
		return errors.New("a receiver that loaded an artifact set must record its own duration")
	}
	log, ok := build["log"].(map[string]any)
	if !ok {
		return errors.New("a receiver that loaded an artifact set must record its own log")
	}
	if err := closedFields(log, buildLogFields, nil, "build.log"); err != nil {
		return err
	}
	if digest, ok := log["sha256"].(string); !ok || !isHex64(digest) {
		return errors.New("build.log sha256 is not a lowercase 64-character digest")
	}
	if _, err := boundedBuildNumber(
		log["size_bytes"], "build.log.size_bytes", 0, maxBuildSourceBytes,
	); err != nil {
		return err
	}
	copiedLog, err := sameDocument(log, producerEvidence["log"])
	if err != nil {
		return err
	}
	if copiedLog {
		return errors.New("the receiver's log is the producer's log identity")
	}
	return nil
}

func verifyBuild(golden *BuildGolden) error {
	requestDoc, err := buildRequestDocument(golden.Request)
	if err != nil {
		return err
	}
	if err := checkDocument(
		requestDoc, golden.RequestDigestDocument, golden.RequestCanonicalJSON,
		golden.RequestDigest, "build request",
	); err != nil {
		return err
	}
	artifactSet, ok := golden.Request["candidate_digest"].(map[string]any)["value"].(string)
	if !ok || artifactSet != golden.ArtifactSetDigest {
		return errors.New("build artifact-set digest differs from the candidate digest")
	}
	bundleDocument := golden.Request["artifact"].(map[string]any)["bundle"].(map[string]any)
	manifest, err := bundleManifest(
		bundleDocument["schema_id"].(string), bundleDocument["entries"].([]any),
	)
	if err != nil {
		return err
	}
	manifestBytes, err := canonical(manifest)
	if err != nil {
		return err
	}
	if shaHex(manifestBytes) != golden.ArtifactSetDigest {
		return errors.New("artifact-set digest does not bind the canonical manifest")
	}
	frozenManifestBytes, err := canonical(golden.BundleManifest)
	if err != nil {
		return err
	}
	if !bytes.Equal(manifestBytes, frozenManifestBytes) {
		return errors.New("independently constructed build manifest differs")
	}
	build := golden.Request["build"].(map[string]any)
	producer := golden.ProducerRecord
	sameProducer, err := sameDocument(build["producer"], producer)
	if err != nil {
		return err
	}
	if !sameProducer {
		return errors.New("the request does not carry the frozen producer link")
	}
	digest, err := buildProducerDigest(golden.ProducerEvidence)
	if err != nil {
		return err
	}
	if producer["evidence_sha256"] != digest {
		return errors.New("the producer link does not bind the producer evidence document")
	}
	if err := verifyBuildEvidence(
		golden.Evidence, build, golden.Request["artifact"].(map[string]any),
		golden.ProducerEvidence, golden.ArtifactSetDigest,
	); err != nil {
		return err
	}
	evidenceBuild := golden.Evidence["build"].(map[string]any)
	if evidenceBuild["artifacts"].(map[string]any)["artifact_set_digest"] != golden.ArtifactSetDigest {
		return errors.New("evidence build artifacts differ from the artifact set")
	}
	// The build identity is inside the request digest, so the receipt that binds
	// the request binds the build too.
	receiptDoc := map[string]any{
		"document_type":    "tcenv.remote.npu-build.receipt-id",
		"document_version": 5,
		"digest_algorithm": "sha256-rfc8785-v1",
		"request_id":       golden.Receipt.RequestID,
		"attempt_id":       golden.Receipt.AttemptID,
		"request_digest":   golden.RequestDigest,
		"candidate_digest": golden.Request["candidate_digest"],
		"device_group":     golden.Request["execution"].(map[string]any)["device_group"],
	}
	receiptBytes, err := canonical(receiptDoc)
	if err != nil {
		return err
	}
	if shaHex(receiptBytes) != golden.Receipt.ReceiptID {
		return errors.New("build receipt id does not bind the request")
	}
	return nil
}
