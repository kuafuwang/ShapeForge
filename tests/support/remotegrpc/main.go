// Non-Python conformance client for the tcenv.remote.v1 evaluator.
//
// It loads the published descriptor set, speaks real gRPC over TLS/mTLS with a
// per-RPC session grant, computes sha256-rfc8785-v1 request digests and verifies
// the attempt receipt, opaque blob digests and the hmac-sha256-v1 result
// signature without importing Python or generated Go stubs.
package main

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/types/descriptorpb"
	"google.golang.org/protobuf/types/dynamicpb"
)

const (
	apiContractID      = "tcenv.remote.v1.evaluator-api.1"
	evidenceContractID = "tcenv.remote.v1.evidence.1"
	evaluatorMethod    = "/tcenv.remote.v1.Evaluator/"
)

type keyringFile struct {
	SchemaVersion int               `json:"schema_version"`
	Keys          map[string]string `json:"keys"`
	Revoked       []string          `json:"revoked"`
}

func (k *keyringFile) trusted(keyID string) ([]byte, error) {
	for _, revoked := range k.Revoked {
		if revoked == keyID {
			return nil, errors.New("signature key was revoked")
		}
	}
	value, ok := k.Keys[keyID]
	if !ok {
		return nil, errors.New("signature key is not trusted")
	}
	key, err := hex.DecodeString(value)
	if err != nil || len(key) < 32 {
		return nil, errors.New("signature key is invalid")
	}
	return key, nil
}

func loadKeyring(path string) (*keyringFile, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	document := &keyringFile{}
	if err := json.Unmarshal(raw, document); err != nil {
		return nil, err
	}
	if document.SchemaVersion != 1 || len(document.Keys) == 0 {
		return nil, errors.New("keyring schema or keys are invalid")
	}
	return document, nil
}

type descriptors struct {
	files *protoregistry.Files
}

func loadDescriptors(path string) (*descriptors, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	set := &descriptorpb.FileDescriptorSet{}
	if err := proto.Unmarshal(raw, set); err != nil {
		return nil, err
	}
	files, err := protodesc.NewFiles(set)
	if err != nil {
		return nil, err
	}
	return &descriptors{files: files}, nil
}

func (d *descriptors) new(name string) *dynamicpb.Message {
	desc, err := d.files.FindDescriptorByName(protoreflect.FullName(name))
	if err != nil {
		panic(err)
	}
	return dynamicpb.NewMessage(desc.(protoreflect.MessageDescriptor))
}

func field(m protoreflect.Message, name string) protoreflect.FieldDescriptor {
	fd := m.Descriptor().Fields().ByName(protoreflect.Name(name))
	if fd == nil {
		panic("missing field " + name)
	}
	return fd
}

func setString(m protoreflect.Message, name, value string) {
	m.Set(field(m, name), protoreflect.ValueOfString(value))
}

func setBytesValue(m protoreflect.Message, name string, value []byte) {
	m.Set(field(m, name), protoreflect.ValueOfBytes(value))
}

func setUint(m protoreflect.Message, name string, value uint64) {
	fd := field(m, name)
	switch fd.Kind() {
	case protoreflect.Uint32Kind:
		m.Set(fd, protoreflect.ValueOfUint32(uint32(value)))
	case protoreflect.Uint64Kind:
		m.Set(fd, protoreflect.ValueOfUint64(value))
	default:
		panic("setUint on non-unsigned field " + name)
	}
}

func setBool(m protoreflect.Message, name string, value bool) {
	m.Set(field(m, name), protoreflect.ValueOfBool(value))
}

func setEnum(m protoreflect.Message, name string, number protoreflect.EnumNumber) {
	m.Set(field(m, name), protoreflect.ValueOfEnum(number))
}

func setMessage(m protoreflect.Message, name string) protoreflect.Message {
	return m.Mutable(field(m, name)).Message()
}

func setMessageCopy(m protoreflect.Message, name string, source protoreflect.Message) protoreflect.Message {
	target := setMessage(m, name)
	proto.Merge(target.Interface().(proto.Message), source.Interface().(proto.Message))
	return target
}

func appendString(m protoreflect.Message, name, value string) {
	m.Mutable(field(m, name)).List().Append(protoreflect.ValueOfString(value))
}

func getString(m protoreflect.Message, name string) string {
	return m.Get(field(m, name)).String()
}

func getUint(m protoreflect.Message, name string) uint64 {
	return m.Get(field(m, name)).Uint()
}

func getBool(m protoreflect.Message, name string) bool {
	return m.Get(field(m, name)).Bool()
}

func getEnum(m protoreflect.Message, name string) protoreflect.EnumNumber {
	return m.Get(field(m, name)).Enum()
}

func getMessage(m protoreflect.Message, name string) protoreflect.Message {
	return m.Get(field(m, name)).Message()
}

func getBytes(m protoreflect.Message, name string) []byte {
	return append([]byte(nil), m.Get(field(m, name)).Bytes()...)
}

func has(m protoreflect.Message, name string) bool {
	return m.Has(field(m, name))
}

func timestampText(m protoreflect.Message) string {
	seconds := m.Get(field(m, "seconds")).Int()
	nanos := m.Get(field(m, "nanos")).Int()
	millis := nanos / 1_000_000
	return time.Unix(seconds, 0).UTC().Format("2006-01-02T15:04:05") + fmt.Sprintf(".%03dZ", millis)
}

func provenanceMap(m protoreflect.Message) map[string]any {
	measurement := "measured"
	if getEnum(m, "measurement_kind") == 2 {
		measurement = "simulated"
	}
	document := map[string]any{
		"schema_version":       int(getUint(m, "schema_version")),
		"evaluator_epoch":      getString(m, "evaluator_epoch"),
		"site_id":              getString(m, "site_id"),
		"device_id":            getString(m, "device_id"),
		"backend":              getString(m, "backend"),
		"backend_version":      getString(m, "backend_version"),
		"target_id":            getString(m, "target_id"),
		"worker_build":         getString(m, "worker_build"),
		"task_pack_digest":     getString(m, "task_pack_digest"),
		"calibration_digest":   getString(m, "calibration_digest"),
		"protocol_version":     getString(m, "protocol_version"),
		"api_contract_id":      getString(m, "api_contract_id"),
		"evidence_contract_id": getString(m, "evidence_contract_id"),
		"measurement_kind":     measurement,
		"publication_allowed":  getBool(m, "publication_allowed"),
	}
	if has(m, "attempt_id") {
		document["attempt_id"] = getString(m, "attempt_id")
	}
	if has(m, "request_digest") {
		document["request_digest"] = getString(m, "request_digest")
	}
	return document
}

func receiptMap(m protoreflect.Message) map[string]any {
	return map[string]any{
		"receipt_id":       getString(m, "receipt_id"),
		"request_id":       getString(m, "request_id"),
		"attempt_id":       getString(m, "attempt_id"),
		"request_digest":   getString(m, "request_digest"),
		"candidate_sha256": getString(m, "candidate_sha256"),
		"lease_fence":      strconv.FormatUint(getUint(m, "lease_fence"), 10),
		"session_sequence": strconv.FormatUint(getUint(m, "session_sequence"), 10),
		"issued_at":        timestampText(getMessage(m, "issued_at")),
	}
}

func blobMap(m protoreflect.Message) map[string]any {
	return map[string]any{
		"schema_id":  getString(m, "schema_id"),
		"media_type": getString(m, "media_type"),
		"size_bytes": int(getUint(m, "size_bytes")),
		"sha256":     getString(m, "sha256"),
	}
}

func checkBlob(descriptor map[string]any, data []byte, name string) error {
	if descriptor["size_bytes"].(int) != len(data) {
		return fmt.Errorf("%s size does not match exact bytes", name)
	}
	if descriptor["sha256"] != shaHex(data) {
		return fmt.Errorf("%s SHA-256 does not match exact bytes", name)
	}
	return nil
}

func verifyResult(d *descriptors, result protoreflect.Message, keyring *keyringFile, siteID, candidateSHA string) (string, error) {
	provenance := provenanceMap(getMessage(result, "provenance"))
	receiptMessage := getMessage(result, "receipt")
	issuedAt := getMessage(receiptMessage, "issued_at")
	if issuedAt.Get(field(issuedAt, "nanos")).Int()%1_000_000 != 0 {
		return "", errors.New("receipt issued_at has sub-millisecond precision")
	}
	receipt := receiptMap(receiptMessage)
	reportData := getBytes(getMessage(result, "report"), "data")
	reportDescriptor := blobMap(getMessage(result, "report"))
	if err := checkBlob(reportDescriptor, reportData, "report"); err != nil {
		return "", err
	}
	var evidenceDescriptor any
	if has(result, "evidence") {
		data := getBytes(getMessage(result, "evidence"), "data")
		descriptor := blobMap(getMessage(result, "evidence"))
		if err := checkBlob(descriptor, data, "evidence"); err != nil {
			return "", err
		}
		evidenceDescriptor = descriptor
	}
	if provenance["attempt_id"] != receipt["attempt_id"] || provenance["request_digest"] != receipt["request_digest"] {
		return "", errors.New("provenance and receipt attempt/digest differ")
	}
	if receipt["candidate_sha256"] != candidateSHA {
		return "", errors.New("receipt candidate digest does not match the request")
	}
	expectedReceipt := shaHex(mustCanonical(receiptDocument(
		receipt["request_id"].(string), receipt["attempt_id"].(string), receipt["request_digest"].(string),
	)))
	if receipt["receipt_id"] != expectedReceipt {
		return "", errors.New("receipt id does not bind the attempt")
	}
	signature := getMessage(result, "signature")
	keyID := getString(signature, "key_id")
	if getString(signature, "algorithm") != "hmac-sha256-v1" {
		return "", errors.New("unsupported result signature algorithm")
	}
	if !strings.HasPrefix(keyID, siteID+"/") {
		return "", errors.New("result key id is not scoped to the evaluator site")
	}
	rawSignature := getBytes(signature, "signature")
	if len(rawSignature) != 32 {
		return "", errors.New("result signature is not 32 bytes")
	}
	key, err := keyring.trusted(keyID)
	if err != nil {
		return "", err
	}
	document, err := resultDocument(map[string]any{
		"provenance": provenance,
		"report":     reportDescriptor,
		"evidence":   evidenceDescriptor,
		"receipt":    receipt,
	}, keyID)
	if err != nil {
		return "", err
	}
	payload := mustCanonical(document)
	mac := hmac.New(sha256.New, key)
	mac.Write(payload)
	if !hmac.Equal(mac.Sum(nil), rawSignature) {
		return "", errors.New("result signature does not verify")
	}
	return receipt["attempt_id"].(string), nil
}

func mustCanonical(document map[string]any) []byte {
	encoded, err := canonical(document)
	if err != nil {
		panic(err)
	}
	return encoded
}

type expectedEvaluator struct {
	SiteID         string
	DeviceID       string
	TargetID       string
	WorkerBuild    string
	TaskPackDigest string
	EvaluatorEpoch string
}

func (e expectedEvaluator) fill(m protoreflect.Message) {
	setString(m, "site_id", e.SiteID)
	setString(m, "device_id", e.DeviceID)
	setString(m, "target_id", e.TargetID)
	setString(m, "worker_build", e.WorkerBuild)
	setString(m, "task_pack_digest", e.TaskPackDigest)
	setString(m, "evaluator_epoch", e.EvaluatorEpoch)
}

func expectedFromProvenance(m protoreflect.Message) expectedEvaluator {
	return expectedEvaluator{
		SiteID:         getString(m, "site_id"),
		DeviceID:       getString(m, "device_id"),
		TargetID:       getString(m, "target_id"),
		WorkerBuild:    getString(m, "worker_build"),
		TaskPackDigest: getString(m, "task_pack_digest"),
		EvaluatorEpoch: getString(m, "evaluator_epoch"),
	}
}

func (e expectedEvaluator) identities(d *descriptors) protoreflect.Message {
	identities := d.new("tcenv.remote.v1.ExpectedIdentities").ProtoReflect()
	e.fill(setMessage(identities, "evaluator"))
	setString(identities, "api_contract_id", apiContractID)
	setString(identities, "evidence_contract_id", evidenceContractID)
	return identities
}

type requestOptions struct {
	Kind               string
	Mode               string
	RunID              string
	SessionID          string
	RequestID          string
	Task               string
	Dtype              string
	Size               string
	TargetID           string
	TaskFingerprint    string
	Source             []byte
	Seed               string
	SeedPresent        bool
	Reps               int
	QueueTimeoutMs     int
	CandidateTimeoutMs int
	FastBench          bool
	Validate           bool
	MeasureSerial      bool
	CachePolicy        string
}

func (o requestOptions) digestDocument() map[string]any {
	var seed any
	if o.SeedPresent {
		seed = o.Seed
	}
	cache := map[string]int{"reuse": 1, "refresh": 2, "disabled": 3}[o.CachePolicy]
	if cache == 0 {
		cache = 1
	}
	return map[string]any{
		"kind":                    o.Kind,
		"run_id":                  o.RunID,
		"session_id":              o.SessionID,
		"task":                    o.Task,
		"dtype":                   o.Dtype,
		"size":                    o.Size,
		"target_id":               o.TargetID,
		"task_fingerprint":        o.TaskFingerprint,
		"candidate_sha256":        shaHex(o.Source),
		"mode":                    o.Mode,
		"seed":                    seed,
		"reps":                    o.Reps,
		"queue_timeout_ms":        o.QueueTimeoutMs,
		"candidate_timeout_ms":    o.CandidateTimeoutMs,
		"cases":                   nil,
		"fast_bench":              o.FastBench,
		"require_contract_launch": false,
		"validate":                o.Validate,
		"measure_serial":          o.MeasureSerial,
		"cache_policy":            o.CachePolicy,
	}
}

func (o requestOptions) digest() string {
	document, err := requestDocument(o.digestDocument())
	if err != nil {
		panic(err)
	}
	return shaHex(mustCanonical(document))
}

func (o requestOptions) message(d *descriptors) *dynamicpb.Message {
	m := d.new("tcenv.remote.v1.EvaluationRequest")
	kinds := map[string]protoreflect.EnumNumber{"formal_eval": 1, "selfcheck": 2}
	modes := map[string]protoreflect.EnumNumber{"quick": 1, "timed": 2, "full": 3, "validate": 4}
	caches := map[string]protoreflect.EnumNumber{"reuse": 1, "refresh": 2, "disabled": 3}
	msg := m.ProtoReflect()
	setEnum(msg, "kind", kinds[o.Kind])
	setString(msg, "request_id", o.RequestID)
	setString(msg, "run_id", o.RunID)
	setString(msg, "session_id", o.SessionID)
	setString(msg, "task", o.Task)
	setString(msg, "dtype", o.Dtype)
	setString(msg, "size", o.Size)
	setString(msg, "target_id", o.TargetID)
	setString(msg, "task_fingerprint", o.TaskFingerprint)
	setString(msg, "candidate_sha256", shaHex(o.Source))
	setBytesValue(msg, "source", o.Source)
	setEnum(msg, "mode", modes[o.Mode])
	if o.SeedPresent {
		setString(msg, "seed", o.Seed)
	}
	setUint(msg, "reps", uint64(o.Reps))
	setUint(msg, "queue_timeout_ms", uint64(o.QueueTimeoutMs))
	setUint(msg, "candidate_timeout_ms", uint64(o.CandidateTimeoutMs))
	setEnum(msg, "cache_policy", caches[o.CachePolicy])
	setBool(msg, "fast_bench", o.FastBench)
	setBool(msg, "validate", o.Validate)
	setBool(msg, "measure_serial", o.MeasureSerial)
	return m
}

type conformance struct {
	ctx      context.Context
	conn     *grpc.ClientConn
	d        *descriptors
	token    string
	keyring  *keyringFile
	timeout  time.Duration
	features []string
	expected expectedEvaluator
	report   map[string]any
}

func (c *conformance) step(name string, ok bool, detail string, extra map[string]any) {
	entry := map[string]any{"name": name, "ok": ok}
	if detail != "" {
		entry["detail"] = detail
	}
	for key, value := range extra {
		entry[key] = value
	}
	steps := c.report["steps"].([]map[string]any)
	c.report["steps"] = append(steps, entry)
	if !ok {
		c.report["ok"] = false
	}
}

func (c *conformance) rpcCtx() (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithTimeout(c.ctx, c.timeout)
	return metadata.AppendToOutgoingContext(ctx, "authorization", "Bearer "+c.token), cancel
}

func (c *conformance) capabilities(features []string) (*dynamicpb.Message, error) {
	request := c.d.new("tcenv.remote.v1.CapabilitiesRequest")
	msg := request.ProtoReflect()
	appendString(msg, "api_contract_ids", apiContractID)
	appendString(msg, "evidence_contract_ids", evidenceContractID)
	for _, feature := range features {
		appendString(msg, "required_features", feature)
	}
	setString(msg, "client_nonce", fmt.Sprintf("%032x", time.Now().UnixNano()))
	c.expected.fill(setMessage(msg, "expected"))
	ctx, cancel := c.rpcCtx()
	defer cancel()
	response := c.d.new("tcenv.remote.v1.Capabilities")
	if err := c.conn.Invoke(ctx, evaluatorMethod+"GetCapabilities", request, response); err != nil {
		return nil, err
	}
	return response, nil
}

func (c *conformance) getJob(expected expectedEvaluator, requestID, digest string) (protoreflect.Message, error) {
	request := c.d.new("tcenv.remote.v1.JobRef").ProtoReflect()
	setString(request, "request_id", requestID)
	setString(request, "request_digest", digest)
	setMessageCopy(request, "expected", expected.identities(c.d))
	ctx, cancel := c.rpcCtx()
	defer cancel()
	response := c.d.new("tcenv.remote.v1.JobSnapshot")
	if err := c.conn.Invoke(ctx, evaluatorMethod+"GetJob", request.Interface(), response); err != nil {
		return nil, err
	}
	return response.ProtoReflect(), nil
}

func (c *conformance) submit(options requestOptions, expected expectedEvaluator, capabilityID string) (protoreflect.Message, error) {
	request := c.d.new("tcenv.remote.v1.SubmitRequest").ProtoReflect()
	setMessageCopy(request, "request", options.message(c.d).ProtoReflect())
	setString(request, "request_digest", options.digest())
	setMessageCopy(request, "expected", expected.identities(c.d))
	setString(request, "capability_id", capabilityID)
	for _, feature := range c.features {
		appendString(request, "required_features", feature)
	}
	ctx, cancel := c.rpcCtx()
	defer cancel()
	response := c.d.new("tcenv.remote.v1.JobSnapshot")
	if err := c.conn.Invoke(ctx, evaluatorMethod+"Submit", request.Interface(), response); err != nil {
		return nil, err
	}
	return response.ProtoReflect(), nil
}

func (c *conformance) watchToTerminal(expected expectedEvaluator, requestID, digest string, after uint64, deadline time.Time) (protoreflect.Message, error) {
	if deadline.IsZero() {
		deadline = time.Now().Add(c.timeout * 4)
	}
	for time.Now().Before(deadline) {
		watchCtx, cancel := context.WithDeadline(c.ctx, minTime(deadline, time.Now().Add(c.timeout)))
		stream, err := c.conn.NewStream(
			metadata.AppendToOutgoingContext(watchCtx, "authorization", "Bearer "+c.token),
			&grpc.StreamDesc{StreamName: "Watch", ServerStreams: true},
			evaluatorMethod+"Watch",
		)
		if err != nil {
			cancel()
			return nil, err
		}
		request := c.d.new("tcenv.remote.v1.WatchRequest").ProtoReflect()
		job := setMessage(request, "job")
		setString(job, "request_id", requestID)
		setString(job, "request_digest", digest)
		setMessageCopy(job, "expected", expected.identities(c.d))
		setUint(request, "after_sequence", after)
		if err := stream.SendMsg(request.Interface()); err != nil {
			cancel()
			return nil, err
		}
		_ = stream.CloseSend()
		for {
			event := c.d.new("tcenv.remote.v1.JobEvent")
			err := stream.RecvMsg(event)
			if err == io.EOF {
				break
			}
			if err != nil {
				cancel()
				if status.Code(err) == 14 || status.Code(err) == 4 || status.Code(err) == 1 { // UNAVAILABLE/DEADLINE/CANCELLED
					break
				}
				return nil, err
			}
			eventMsg := event.ProtoReflect()
			if getUint(eventMsg, "sequence") > after {
				after = getUint(eventMsg, "sequence")
			}
			if !has(eventMsg, "snapshot") {
				continue
			}
			snapshot := getMessage(eventMsg, "snapshot")
			state := getEnum(snapshot, "state")
			if state >= 4 {
				cancel()
				return snapshot, nil
			}
		}
		cancel()
		snapshot, err := c.getJob(expected, requestID, digest)
		if err == nil {
			if getUint(snapshot, "last_sequence") > after {
				after = getUint(snapshot, "last_sequence")
			}
			if getEnum(snapshot, "state") >= 4 {
				return snapshot, nil
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	return nil, context.DeadlineExceeded
}

func (c *conformance) retry(expected expectedEvaluator, requestID, digest, attemptID, capabilityID string) (protoreflect.Message, error) {
	request := c.d.new("tcenv.remote.v1.RetryRequest").ProtoReflect()
	job := setMessage(request, "job")
	setString(job, "request_id", requestID)
	setString(job, "request_digest", digest)
	setMessageCopy(job, "expected", expected.identities(c.d))
	setString(request, "interrupted_attempt_id", attemptID)
	setString(request, "capability_id", capabilityID)
	for _, feature := range c.features {
		appendString(request, "required_features", feature)
	}
	ctx, cancel := c.rpcCtx()
	defer cancel()
	response := c.d.new("tcenv.remote.v1.JobSnapshot")
	if err := c.conn.Invoke(ctx, evaluatorMethod+"Retry", request.Interface(), response); err != nil {
		return nil, err
	}
	return response.ProtoReflect(), nil
}

func (c *conformance) cancel(expected expectedEvaluator, requestID, digest string) (protoreflect.Message, error) {
	request := c.d.new("tcenv.remote.v1.CancelRequest").ProtoReflect()
	job := setMessage(request, "job")
	setString(job, "request_id", requestID)
	setString(job, "request_digest", digest)
	setMessageCopy(job, "expected", expected.identities(c.d))
	ctx, cancel := c.rpcCtx()
	defer cancel()
	response := c.d.new("tcenv.remote.v1.JobSnapshot")
	if err := c.conn.Invoke(ctx, evaluatorMethod+"Cancel", request.Interface(), response); err != nil {
		return nil, err
	}
	return response.ProtoReflect(), nil
}

func minTime(a, b time.Time) time.Time {
	if a.Before(b) {
		return a
	}
	return b
}

func runCapabilities(d *descriptors, conn *grpc.ClientConn, token string, features []string, timeout time.Duration) (*dynamicpb.Message, error) {
	client := &conformance{ctx: context.Background(), conn: conn, d: d, token: token, timeout: timeout}
	return client.capabilities(features)
}

func main() {
	descPath := flag.String("desc", "", "evaluator.desc descriptor set")
	endpoint := flag.String("endpoint", "", "host:port")
	ca := flag.String("ca", "", "server CA bundle")
	cert := flag.String("cert", "", "client certificate")
	key := flag.String("key", "", "client private key")
	serverName := flag.String("server-name", "", "TLS server name override")
	tokenFile := flag.String("token-file", "", "session grant token file")
	keyringPath := flag.String("keyring", "", "HMAC verifier keyring")
	targetID := flag.String("target-id", "", "target id")
	siteID := flag.String("site-id", "", "expected site id")
	deviceID := flag.String("device-id", "", "expected device id")
	workerBuild := flag.String("worker-build", "", "expected worker build")
	taskPackDigest := flag.String("task-pack-digest", "", "expected task pack digest")
	expectedEpoch := flag.String("expected-epoch", "", "expected evaluator epoch (optional)")
	runID := flag.String("run-id", "", "run id")
	sessionID := flag.String("session-id", "", "session id")
	requestID := flag.String("request-id", "go-conformance-1", "primary request id")
	task := flag.String("task", "", "task name")
	dtype := flag.String("dtype", "fp16", "dtype")
	size := flag.String("size", "small", "size")
	fingerprint := flag.String("fingerprint", "", "task fingerprint")
	sourceFile := flag.String("source-file", "", "candidate source file")
	mode := flag.String("mode", "quick", "evaluation mode")
	kind := flag.String("kind", "formal_eval", "evaluation kind")
	seed := flag.String("seed", "3", "seed; 'none' means null")
	reps := flag.Int("reps", 2, "reps")
	queueTimeout := flag.Int("queue-timeout-ms", 30000, "queue timeout ms")
	candidateTimeout := flag.Int("candidate-timeout-ms", 120000, "candidate timeout ms")
	cancelSource := flag.String("cancel-source-file", "", "second job source to cancel")
	cancelRequestID := flag.String("cancel-request-id", "go-conformance-cancel-1", "second request id")
	timeout := flag.Duration("timeout", 30*time.Second, "per-RPC timeout")
	out := flag.String("out", "", "write the JSON report here")
	flag.Parse()
	if *descPath == "" || *endpoint == "" || *ca == "" || *cert == "" || *key == "" || *tokenFile == "" || *keyringPath == "" || *targetID == "" || *siteID == "" || *deviceID == "" || *workerBuild == "" || *taskPackDigest == "" || *runID == "" || *sessionID == "" || *task == "" || *fingerprint == "" || *sourceFile == "" {
		fmt.Fprintln(os.Stderr, "missing required flags")
		os.Exit(2)
	}
	report := map[string]any{"schema_version": 1, "client": "go", "ok": true, "steps": []map[string]any{}, "evaluator_endpoint": *endpoint}
	fail := func(err error) {
		report["ok"] = false
		report["error"] = err.Error()
		writeReport(report, *out)
		os.Exit(1)
	}
	tokenBytes, err := os.ReadFile(*tokenFile)
	if err != nil {
		fail(err)
	}
	token := strings.TrimSpace(string(tokenBytes))
	keyring, err := loadKeyring(*keyringPath)
	if err != nil {
		fail(err)
	}
	d, err := loadDescriptors(*descPath)
	if err != nil {
		fail(err)
	}
	caBytes, err := os.ReadFile(*ca)
	if err != nil {
		fail(err)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(caBytes) {
		fail(errors.New("cannot parse CA bundle"))
	}
	certificate, err := tls.LoadX509KeyPair(*cert, *key)
	if err != nil {
		fail(err)
	}
	tlsConfig := &tls.Config{RootCAs: pool, Certificates: []tls.Certificate{certificate}, MinVersion: tls.VersionTLS12}
	if *serverName != "" {
		tlsConfig.ServerName = *serverName
	}
	conn, err := grpc.NewClient(*endpoint, grpc.WithTransportCredentials(credentials.NewTLS(tlsConfig)))
	if err != nil {
		fail(err)
	}
	defer conn.Close()
	source, err := os.ReadFile(*sourceFile)
	if err != nil {
		fail(err)
	}
	options := requestOptions{
		Kind: *kind, Mode: *mode, RunID: *runID, SessionID: *sessionID, RequestID: *requestID,
		Task: *task, Dtype: *dtype, Size: *size, TargetID: *targetID,
		TaskFingerprint: *fingerprint, Source: source, Reps: *reps,
		QueueTimeoutMs: *queueTimeout, CandidateTimeoutMs: *candidateTimeout,
		CachePolicy: "reuse",
	}
	if *seed != "none" {
		options.Seed = *seed
		options.SeedPresent = true
	}
	client := &conformance{
		ctx: context.Background(), conn: conn, d: d, token: token, keyring: keyring,
		timeout: *timeout, features: []string{"durable-jobs", "attempt-receipts", "progress"},
		expected: expectedEvaluator{
			SiteID: *siteID, DeviceID: *deviceID, TargetID: *targetID,
			WorkerBuild: *workerBuild, TaskPackDigest: *taskPackDigest,
			EvaluatorEpoch: *expectedEpoch,
		},
		report: report,
	}

	// 1. Unauthenticated capabilities must be refused.
	unauth := &conformance{ctx: context.Background(), conn: conn, d: d, token: "", timeout: *timeout, report: report}
	if _, err := unauth.capabilities(nil); err == nil {
		client.step("unauthenticated_capabilities_rejected", false, "call succeeded", nil)
	} else {
		client.step("unauthenticated_capabilities_rejected", status.Code(err) == 16, err.Error(), nil)
	}

	// 2. Unknown required feature fails closed with a typed detail.
	unsupported := "go-conformance-unsupported"
	if _, err := client.capabilities([]string{"durable-jobs", unsupported}); err == nil {
		client.step("unsupported_required_feature_rejected", false, "capabilities succeeded", nil)
	} else {
		missing := missingFeatures(d, err)
		client.step("unsupported_required_feature_rejected", status.Code(err) == 9 && contains(missing, unsupported), err.Error(), map[string]any{"missing_features": missing})
	}

	// 3. Capabilities and the three identities.
	capabilities, err := client.capabilities([]string{"durable-jobs", "attempt-receipts", "progress"})
	if err != nil {
		fail(err)
	}
	capMsg := capabilities.ProtoReflect()
	provenance := getMessage(capMsg, "provenance")
	client.step("capabilities", getString(capMsg, "api_contract_id") == apiContractID && getString(capMsg, "evidence_contract_id") == evidenceContractID, "", map[string]any{
		"evaluator_epoch":      getString(capMsg, "evaluator_epoch"),
		"api_contract_id":      getString(capMsg, "api_contract_id"),
		"evidence_contract_id": getString(capMsg, "evidence_contract_id"),
		"site_id":              getString(provenance, "site_id"),
		"target_id":            getString(provenance, "target_id"),
		"measurement_kind":     provenanceMap(provenance)["measurement_kind"],
	})
	expected := expectedFromProvenance(provenance)
	digest := options.digest()

	// 4. Submit + Watch/GetJob, with explicit Retry for an interrupted attempt.
	snapshot, err := client.submit(options, expected, getString(capMsg, "capability_id"))
	if err != nil {
		fail(err)
	}
	snapshot, err = client.watchToTerminal(expected, options.RequestID, digest, getUint(snapshot, "last_sequence"), time.Time{})
	if err != nil {
		fail(err)
	}
	if getEnum(snapshot, "state") == 4 {
		attempt := getString(getMessage(snapshot, "current_attempt"), "attempt_id")
		fresh, err := client.capabilities([]string{"durable-jobs", "attempt-receipts", "progress"})
		if err != nil {
			fail(err)
		}
		freshMsg := fresh.ProtoReflect()
		expected = expectedFromProvenance(getMessage(freshMsg, "provenance"))
		retried, err := client.retry(expected, options.RequestID, digest, attempt, getString(freshMsg, "capability_id"))
		if err != nil {
			fail(err)
		}
		snapshot, err = client.watchToTerminal(expected, options.RequestID, digest, getUint(retried, "last_sequence"), time.Time{})
		if err != nil {
			fail(err)
		}
		client.step("retry_interrupted_attempt", getEnum(snapshot, "state") == 5, "", map[string]any{"attempt_id": attempt})
	}
	client.step("submit_watch_result", getEnum(snapshot, "state") == 5, "", nil)

	// 5. GetJob agrees and the result verifies independently.
	finalSnapshot, err := client.getJob(expected, options.RequestID, digest)
	if err != nil {
		fail(err)
	}
	client.step("getjob_terminal_snapshot", getEnum(finalSnapshot, "state") == 5, "", map[string]any{"last_sequence": getUint(finalSnapshot, "last_sequence")})
	verifiedAttempt := ""
	if has(finalSnapshot, "result") {
		verifiedAttempt, err = verifyResult(d, getMessage(finalSnapshot, "result"), keyring, expected.SiteID, shaHex(source))
		if err != nil {
			client.step("verify_result_signature", false, err.Error(), nil)
		} else {
			client.step("verify_result_signature", true, "", map[string]any{"attempt_id": verifiedAttempt})
		}
	} else {
		client.step("verify_result_signature", false, "terminal snapshot has no result", nil)
	}

	// 6. Retry of a completed job is refused.
	fresh, err := client.capabilities([]string{"durable-jobs", "attempt-receipts", "progress"})
	if err == nil {
		freshMsg := fresh.ProtoReflect()
		freshExpected := expectedFromProvenance(getMessage(freshMsg, "provenance"))
		_, err = client.retry(freshExpected, options.RequestID, digest, verifiedAttempt, getString(freshMsg, "capability_id"))
	}
	client.step("retry_completed_rejected", err != nil && status.Code(err) == 9, errorText(err), nil)

	// 7. Cancel a second job.
	if *cancelSource != "" {
		secondSource, readErr := os.ReadFile(*cancelSource)
		if readErr != nil {
			fail(readErr)
		}
		second := options
		second.RequestID = *cancelRequestID
		second.Source = secondSource
		second.CandidateTimeoutMs = 20000
		caps, err := client.capabilities([]string{"durable-jobs", "attempt-receipts", "progress"})
		if err != nil {
			fail(err)
		}
		capsMsg := caps.ProtoReflect()
		secondExpected := expectedFromProvenance(getMessage(capsMsg, "provenance"))
		submitted, err := client.submit(second, secondExpected, getString(capsMsg, "capability_id"))
		if err != nil {
			fail(err)
		}
		if _, cancelErr := client.cancel(secondExpected, second.RequestID, second.digest()); cancelErr != nil {
			client.step("cancel_second_job_request", false, cancelErr.Error(), nil)
		}
		final, err := client.getJob(secondExpected, second.RequestID, second.digest())
		if err != nil {
			fail(err)
		}
		deadline := time.Now().Add(10 * time.Second)
		for getEnum(final, "state") < 5 && time.Now().Before(deadline) {
			time.Sleep(100 * time.Millisecond)
			final, err = client.getJob(secondExpected, second.RequestID, second.digest())
			if err != nil {
				break
			}
		}
		state := int(getEnum(final, "state"))
		client.step("cancel_second_job", state == 7 || state == 5, "", map[string]any{"state": state, "submitted_state": int(getEnum(submitted, "state"))})
	}
	writeReport(report, *out)
	if report["ok"] != true {
		os.Exit(1)
	}
}

func writeReport(report map[string]any, path string) {
	encoded, _ := json.MarshalIndent(report, "", "  ")
	if path != "" {
		_ = os.WriteFile(path, append(encoded, '\n'), 0o600)
	}
	fmt.Println(string(encoded))
}

func errorText(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func contains(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}

func missingFeatures(d *descriptors, err error) []string {
	st, ok := status.FromError(err)
	if !ok {
		return nil
	}
	for _, detail := range st.Proto().Details {
		if !strings.HasSuffix(detail.TypeUrl, "tcenv.remote.v1.EvaluationErrorDetail") {
			continue
		}
		message := d.new("tcenv.remote.v1.EvaluationErrorDetail")
		if proto.Unmarshal(detail.Value, message) != nil {
			return nil
		}
		list := message.ProtoReflect().Get(field(message.ProtoReflect(), "missing_features")).List()
		result := make([]string, 0, list.Len())
		for index := 0; index < list.Len(); index++ {
			result = append(result, list.Get(index).String())
		}
		return result
	}
	return nil
}
