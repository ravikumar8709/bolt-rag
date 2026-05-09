#!/bin/bash

# Production Deployment Script for RAG Question Answer System

set -e

echo "🚀 Starting production deployment..."

# Check if .env.production exists
if [ ! -f ".env.production" ]; then
    echo "❌ .env.production file not found!"
    echo "📝 Please copy .env.production.template to .env.production and fill in your values:"
    echo "   cp .env.production.template .env.production"
    echo "   nano .env.production"
    exit 1
fi

# Load environment variables
set -a
source .env.production
set +a

# Check required environment variables
required_vars=("POSTGRES_PASSWORD" "PINECONE_API_KEY" "PINECONE_INDEX_NAME" "GROQ_API_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Required environment variable $var is not set in .env.production"
        exit 1
    fi
done

echo "✅ Environment variables validated"

# Create SSL directory if it doesn't exist
if [ ! -d "ssl" ]; then
    echo "📁 Creating SSL directory..."
    mkdir -p ssl
    echo "⚠️  Please add your SSL certificates to the ssl/ directory:"
    echo "   - ssl/cert.pem (SSL certificate)"
    echo "   - ssl/key.pem (SSL private key)"
    echo "   Or generate self-signed certificates for testing:"
    echo "   openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes"
fi

# Build and start services
echo "🔨 Building Docker images..."
docker-compose -f docker-compose.prod.yml build --no-cache

echo "🗄️  Starting database..."
docker-compose -f docker-compose.prod.yml up -d db

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
sleep 10

# Run database migrations
echo "🔄 Running database migrations..."
docker-compose -f docker-compose.prod.yml exec -T db psql -U rag_user -d rag_db -c "SELECT 1;" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Database is ready"
else
    echo "❌ Database is not ready. Please check the logs:"
    echo "   docker-compose -f docker-compose.prod.yml logs db"
    exit 1
fi

# Start all services
echo "🚀 Starting all services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for application to be ready
echo "⏳ Waiting for application to be ready..."
sleep 15

# Health check
echo "🏥 Performing health check..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Application is healthy and running!"
    echo "🌐 Application is available at:"
    echo "   - HTTP: http://localhost"
    echo "   - HTTPS: https://localhost (if SSL is configured)"
    echo "   - Direct API: http://localhost:8000"
else
    echo "❌ Health check failed. Please check the logs:"
    echo "   docker-compose -f docker-compose.prod.yml logs app"
    exit 1
fi

echo "📊 To view logs:"
echo "   docker-compose -f docker-compose.prod.yml logs -f"
echo "🛑 To stop services:"
echo "   docker-compose -f docker-compose.prod.yml down"
echo "🔄 To restart services:"
echo "   docker-compose -f docker-compose.prod.yml restart"

echo "🎉 Deployment completed successfully!"