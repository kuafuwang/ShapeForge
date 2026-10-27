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
