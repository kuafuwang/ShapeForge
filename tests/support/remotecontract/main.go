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
		if strings.HasPrefix(path, "/") || strings.Contains(path, "\\") ||
			strings.HasSuffix(path, "/") || strings.Contains(path, "//") {
			return nil, fmt.Errorf("unsafe bundle path %q", path)
		}
		for _, segment := range strings.Split(path, "/") {
			if segment == "" || segment == "." || segment == ".." {
				return nil, fmt.Errorf("unsafe bundle path segment in %q", path)
			}
		}
		if seen[strings.ToLower(path)] {
			return nil, fmt.Errorf("duplicate bundle path %q", path)
		}
		seen[strings.ToLower(path)] = true
		if _, ok := canonicalEntry["size_bytes"].(json.Number); !ok {
			return nil, fmt.Errorf("bundle entry %q size is not an integer", path)
		}
		if digest, ok := canonicalEntry["sha256"].(string); !ok || len(digest) != 64 {
			return nil, fmt.Errorf("bundle entry %q SHA-256 is invalid", path)
		}
		manifest = append(manifest, canonicalEntry)
	}
	if len(manifest) == 0 {
		return nil, errors.New("bundle has no entries")
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
	for _, raw := range rawDevices {
		name, ok := raw.(string)
		if !ok || name == "" {
			return errors.New("declared device id is not text")
		}
		declared = append(declared, name)
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
	// -1 marks "this dimension does not exist for this vendor shape".
	coordinateValue := func(span map[string]any, name string) (int64, error) {
		raw := span[name]
		if raw == nil {
			return -1, nil
		}
		value, ok := raw.(json.Number)
		if !ok {
			return 0, fmt.Errorf("span %s is not an integer", name)
		}
		return strconv.ParseInt(value.String(), 10, 64)
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
	timing := golden.Evidence["timing"].(map[string]any)
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
