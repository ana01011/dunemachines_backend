#!/bin/bash

echo "🧪 SARAH AI SYSTEM TEST"
echo "======================="
echo

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

TESTS_PASSED=0
TESTS_FAILED=0

# Test 1: Health Check
echo -n "1. Health Check: "
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ FAIL${NC}"
    ((TESTS_FAILED++))
fi

# Test 2: Basic Chat
echo -n "2. Basic Chat: "
RESPONSE=$(curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}')
if echo "$RESPONSE" | grep -q "response"; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ FAIL${NC}"
    ((TESTS_FAILED++))
fi

# Test 3: Creator Attribution
echo -n "3. Creator Attribution: "
RESPONSE=$(curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Who created you?"}')
if echo "$RESPONSE" | grep -q "Ahmed"; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ FAIL${NC}"
    ((TESTS_FAILED++))
fi

# Test 4: Memory Storage
USER="memtest_$(date +%s)"
echo -n "4. Memory Storage: "
curl -s -X POST http://localhost:8000/api/chat/with-memory \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"My favorite color is purple\", \"user_id\": \"$USER\"}" > /dev/null

RESPONSE=$(curl -s http://localhost:8000/api/memory/$USER)
if echo "$RESPONSE" | grep -q "purple"; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ FAIL${NC}"
    ((TESTS_FAILED++))
fi

# Test 5: Memory Recall
echo -n "5. Memory Recall: "
RESPONSE=$(curl -s -X POST http://localhost:8000/api/chat/with-memory \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What is my favorite color?\", \"user_id\": \"$USER\"}")
if echo "$RESPONSE" | grep -i -q "purple"; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠️  PARTIAL${NC} (Memory stored but not recalled)"
    ((TESTS_FAILED++))
fi

# Test 6: Performance Endpoint
echo -n "6. Performance Stats: "
if curl -s http://localhost:8000/api/performance | grep -q "cpu"; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ FAIL${NC}"
    ((TESTS_FAILED++))
fi

# Test 7: Cache System
echo -n "7. Cache System: "
MSG="What is the capital of France?"
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"$MSG\"}" > /dev/null
sleep 1
RESPONSE=$(curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"$MSG\"}")
if echo "$RESPONSE" | grep -q '"cached":true'; then
    echo -e "${GREEN}✅ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠️  Cache not working${NC}"
    ((TESTS_FAILED++))
fi

# Summary
echo
echo "======================="
echo -e "Results: ${GREEN}$TESTS_PASSED Passed${NC}, ${RED}$TESTS_FAILED Failed${NC}"
echo

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All systems operational!${NC}"
else
    echo -e "${YELLOW}⚠️  Some features need attention${NC}"
fi

# Show current stats
echo
echo "📊 Current System Stats:"
curl -s http://localhost:8000/api/performance | jq '{
  cpu_usage: .cpu.percent,
  memory_usage: .memory.percent,
  cache_entries: .cache.entries,
  active_sessions: .cache.memory_sessions
}'
