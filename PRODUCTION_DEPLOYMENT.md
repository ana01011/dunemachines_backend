# Sarah AI Production Deployment Guide

## 🚀 Production Ready Components Analysis

### ✅ Working Components

#### 1. **Core Application**
- **Main Application**: `main.py` - FastAPI-based REST API server
- **Framework**: FastAPI with async/await support
- **AI Model**: Mistral 7B with Llama.cpp integration
- **Database**: PostgreSQL with asyncpg
- **Authentication**: JWT-based auth with bcrypt password hashing
- **WebSocket**: Real-time chat support via WebSocket

#### 2. **Database Layer**
- **Schema**: Complete PostgreSQL schema with tables for:
  - Users and authentication
  - User profiles and preferences
  - Conversations and messages
  - User memories and facts
  - Relationship tracking
- **Migrations**: SQL schema file available at `scripts/init_database.sql`
- **Connection Pooling**: Async connection pooling with asyncpg
- **Indexes**: Optimized indexes for performance

#### 3. **API Endpoints**
- `/api/v1/auth/*` - Authentication (login, register, logout)
- `/api/v1/chat/*` - Chat functionality
- `/api/v1/users/*` - User management
- `/api/v1/themes/*` - Theme management
- `/api/v1/omnius/*` - Advanced AI features (optional)
- `/api/chat` - Direct chat endpoint
- `/api/performance` - Performance monitoring

#### 4. **Security Features**
- JWT token-based authentication
- Password hashing with bcrypt
- CORS middleware configured
- Session management
- Email verification support

#### 5. **AI Features**
- Multiple personality modes
- Context-aware responses
- Memory system for user interactions
- Response caching for performance
- Neurochemistry simulation (experimental)

### 📦 Required Files for Production

#### 1. **Environment Configuration**
✅ Created `.env` file with production settings
- Database credentials
- JWT secrets (MUST BE CHANGED)
- Model paths
- SMTP configuration

#### 2. **Dependencies**
✅ `requirements.txt` - Python dependencies
- Additional system dependencies needed:
  - PostgreSQL 13+
  - Python 3.11+
  - GCC/G++ for llama-cpp compilation

#### 3. **Database**
- PostgreSQL database required
- Run migration script: `scripts/init_database.sql`

#### 4. **AI Model**
- Need to download Mistral 7B model (GGUF format)
- Place in project directory
- Update MODEL_PATH in .env

## 🔧 Deployment Steps

### Step 1: System Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip postgresql postgresql-contrib nginx

# Install build tools for llama-cpp
sudo apt install -y build-essential cmake
```

### Step 2: Database Setup
```bash
# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE sarah_ai_fresh;
CREATE USER sarah_user WITH PASSWORD 'your_secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE sarah_ai_fresh TO sarah_user;
EOF

# Run migrations
cd /workspace
PGPASSWORD=your_secure_password_here psql -h localhost -U sarah_user -d sarah_ai_fresh < scripts/init_database.sql
```

### Step 3: Application Setup
```bash
# Clone repository (if needed)
cd /opt
git clone <repository_url> sarah-ai
cd sarah-ai

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download AI model
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf

# Update .env file
# IMPORTANT: Change JWT_SECRET and database password
nano .env
```

### Step 4: Create Systemd Service
```bash
# Create service file
sudo nano /etc/systemd/system/sarah-ai.service
```

Add the following content:
```ini
[Unit]
Description=Sarah AI Backend Service
After=network.target postgresql.service

[Service]
Type=exec
User=www-data
WorkingDirectory=/opt/sarah-ai
Environment="PATH=/opt/sarah-ai/venv/bin"
ExecStart=/opt/sarah-ai/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/sarah-ai/app.log
StandardError=append:/var/log/sarah-ai/error.log

# Performance optimization
Nice=-20
CPUSchedulingPolicy=rr
CPUSchedulingPriority=99
IOSchedulingClass=realtime
IOSchedulingPriority=0

[Install]
WantedBy=multi-user.target
```

### Step 5: Nginx Configuration
```bash
# Create Nginx configuration
sudo nano /etc/nginx/sites-available/sarah-ai
```

Add:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/sarah-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Step 6: Start Services
```bash
# Create log directory
sudo mkdir -p /var/log/sarah-ai
sudo chown www-data:www-data /var/log/sarah-ai

# Start and enable service
sudo systemctl daemon-reload
sudo systemctl start sarah-ai
sudo systemctl enable sarah-ai

# Check status
sudo systemctl status sarah-ai

# View logs
sudo journalctl -u sarah-ai -f
```

## 🔍 Verification Checklist

### Database Verification
```bash
# Test database connection
PGPASSWORD=your_password psql -h localhost -U sarah_user -d sarah_ai_fresh -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';"
```

### API Health Check
```bash
# Check if API is running
curl http://localhost:8000/health

# Check API info
curl http://localhost:8000/

# Test performance endpoint
curl http://localhost:8000/api/performance
```

### Test Chat Functionality
```bash
# Test chat endpoint
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, Sarah!"}'
```

## 🔐 Security Checklist

- [ ] Change JWT_SECRET to a secure random value
- [ ] Change database password
- [ ] Configure firewall (ufw or iptables)
- [ ] Set up SSL/TLS with Let's Encrypt
- [ ] Configure rate limiting in Nginx
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log rotation
- [ ] Set up backup strategy for database

## 📊 Performance Optimization

### CPU Optimization
- Use taskset to pin process to specific cores
- Set nice priority to -20 for better performance
- Configure thread count based on CPU cores

### Database Optimization
- Tune PostgreSQL settings (shared_buffers, work_mem)
- Regular VACUUM and ANALYZE
- Monitor slow queries

### Model Optimization
- Adjust batch size and context size
- Use GPU layers if available
- Enable memory mapping (use_mmap=True)

## 🚨 Monitoring

### Application Logs
```bash
# View application logs
tail -f /var/log/sarah-ai/app.log

# View error logs
tail -f /var/log/sarah-ai/error.log
```

### System Monitoring
```bash
# Monitor CPU and memory
htop

# Monitor network connections
netstat -tulpn | grep 8000

# Database connections
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity WHERE datname = 'sarah_ai_fresh';"
```

## 🔄 Backup and Recovery

### Database Backup Script
```bash
#!/bin/bash
# Save as /opt/sarah-ai/backup.sh

BACKUP_DIR="/opt/sarah-ai/backups"
DB_NAME="sarah_ai_fresh"
DB_USER="sarah_user"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR
pg_dump -U $DB_USER -d $DB_NAME > "$BACKUP_DIR/backup_$TIMESTAMP.sql"

# Keep only last 7 days of backups
find $BACKUP_DIR -type f -mtime +7 -delete
```

### Automated Backups
```bash
# Add to crontab
crontab -e
# Add: 0 2 * * * /opt/sarah-ai/backup.sh
```

## 📝 Environment Variables Reference

| Variable | Description | Production Value |
|----------|-------------|------------------|
| POSTGRES_HOST | Database host | localhost |
| POSTGRES_DB | Database name | sarah_ai_fresh |
| POSTGRES_USER | Database user | sarah_user |
| POSTGRES_PASSWORD | Database password | [CHANGE THIS] |
| JWT_SECRET | JWT signing secret | [GENERATE NEW] |
| MODEL_PATH | Path to AI model | /opt/sarah-ai/mistral-7b.gguf |
| API_PORT | API server port | 8000 |
| ENVIRONMENT | Environment type | production |

## 🆘 Troubleshooting

### Common Issues

1. **Model not found**
   - Verify MODEL_PATH in .env
   - Ensure model file exists and is readable

2. **Database connection failed**
   - Check PostgreSQL is running
   - Verify credentials in .env
   - Check pg_hba.conf for authentication

3. **Out of memory**
   - Reduce MODEL_CONTEXT_SIZE
   - Reduce MODEL_BATCH_SIZE
   - Add swap space if needed

4. **Slow responses**
   - Check CPU usage with htop
   - Verify thread configuration
   - Enable response caching

## 📞 Support Resources

- Documentation: `README.md`
- Architecture: `ARCHITECTURE.md`
- GitHub Actions: `.github/workflows/deploy.yml`
- Production Config: `production_config.md`

## ✅ Production Readiness Summary

The application is production-ready with:
- ✅ Complete backend API
- ✅ Database schema and migrations
- ✅ Authentication system
- ✅ WebSocket support
- ✅ Performance optimizations
- ✅ Deployment automation (GitHub Actions)
- ✅ Monitoring endpoints
- ✅ Error handling

Required actions before deployment:
1. Generate secure JWT_SECRET
2. Set strong database password
3. Download AI model
4. Configure SSL/TLS
5. Set up monitoring
6. Configure backups