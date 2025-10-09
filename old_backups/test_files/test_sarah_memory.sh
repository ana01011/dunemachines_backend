#!/bin/bash

echo "🧪 Testing Sarah with Memory & Context"
echo "======================================"

API_URL="http://localhost:8000/api/v1/chat/chat"
USER_ID="sarah_test_$(date +%s)"

# Test 1: Introduction
echo -e "\n📝 Test 1: Introduction"
curl -s -X POST $API_URL \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "{\"content\": \"Hi Sarah, I'm David and I love coding\", \"personality\": \"sarah\"}" | jq -r .message

sleep 2

# Test 2: Memory check
echo -e "\n📝 Test 2: Testing Memory"
curl -s -X POST $API_URL \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "{\"content\": \"What did I just tell you about myself?\", \"personality\": \"sarah\"}" | jq -r .message

sleep 2

# Test 3: Creator check
echo -e "\n📝 Test 3: Testing Creator Info"
curl -s -X POST $API_URL \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "{\"content\": \"Who created you?\", \"personality\": \"sarah\"}" | jq -r .message

echo -e "\n✅ Test complete!"
