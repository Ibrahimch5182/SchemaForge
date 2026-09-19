#!/usr/bin/env bash
# Post-deploy verification, run ON THE HOST (or from anywhere with curl, using the public URL):
#   bash deploy/aws/verify_stack.sh https://api.example.com
# Checks the public surface only: HTTPS health, model readiness (no inference), and that the private
# model server is NOT reachable from the internet.
set -uo pipefail
BASE="${1:?usage: verify_stack.sh https://your-domain}"
HOST="${BASE#https://}"; HOST="${HOST%%/*}"
fail=0
ok()   { echo "  [PASS] $1"; }
bad()  { echo "  [FAIL] $1"; fail=1; }

code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$BASE/health"); [ "$code" = 200 ] && ok "GET /health -> 200" || bad "GET /health -> $code"
ready=$(curl -s --max-time 15 "$BASE/ready"); echo "  /ready: $ready"
echo "$ready" | grep -q '"ready":true' && ok "model server ready" || bad "not ready (model may still be loading; retry)"
curl -s --max-time 15 "$BASE/databases" | grep -q '"demo"' && ok "demo database registered" || bad "databases"
[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$BASE/docs")" = 404 ] && ok "/docs hidden in production" || bad "/docs exposed"
for port in 8080 8000; do
  if timeout 5 bash -c "exec 3<>/dev/tcp/$HOST/$port" 2>/dev/null; then bad "port $port is open to the internet"; else ok "port $port closed"; fi
done
exit $fail
