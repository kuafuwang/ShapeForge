// Canonical JSON, request digest, receipt id and result-signature document.
//
// The algorithm is copied from the independent Go golden verifier
// (tests/support/remotecontract) so the conformance client computes the same
// sha256-rfc8785-v1 documents as the Python implementation and the golden.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math/big"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

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
