#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to copy logs for a specific service
copy_service_logs() {
    local service_name=$1
    local container_name=$2
    local log_file="./logs/${service_name}.log"
    
    print_info "Copying logs for service: $service_name (container: $container_name)"
    
    # Create logs directory if it doesn't exist
    mkdir -p ./logs
    
    # Check if container exists and is running
    if ! docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
        print_warning "Container $container_name is not running. Checking if it exists..."
        if ! docker ps -a --format "{{.Names}}" | grep -q "^${container_name}$"; then
            print_error "Container $container_name does not exist"
            return 1
        fi
    fi
    
    # Copy logs from Docker to local logs directory
    if docker logs "$container_name" > "$log_file" 2>&1; then
        local log_size=$(wc -c < "$log_file")
        if [ "$log_size" -eq 0 ]; then
            print_warning "No logs found for $service_name"
        else
            print_success "Copied logs for $service_name to $log_file ($(wc -l < "$log_file") lines)"
        fi
    else
        print_error "Failed to copy logs for $service_name"
        return 1
    fi
}

# Function to copy all service logs
copy_all_logs() {
    print_info "Copying logs for all services..."
    
    # Define services and their container names
    local services=(
        "s2s:s2s"
        "lipsync:lipsync"
        "asd:asd"
        "controller:controller"
    )
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r service_name container_name <<< "$service_info"
        
        # Check if container exists before trying to copy logs
        if ! docker ps -a --format "{{.Names}}" | grep -q "^${container_name}$"; then
            print_warning "Container $container_name does not exist, skipping..."
            continue
        fi
        
        copy_service_logs "$service_name" "$container_name"
    done
    
    print_success "All service logs copied to ./logs/ directory"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [SERVICE_NAME]"
    echo ""
    echo "Copy Docker logs from containers to ./logs/ directory"
    echo ""
    echo "SERVICE_NAME: Optional. Copy logs for specific service only."
    echo "              Available services: s2s, lipsync, asd, controller"
    echo ""
    echo "Examples:"
    echo "  $0              # Copy logs for all services"
    echo "  $0 s2s          # Copy logs for S2S service only"
    echo "  $0 lipsync     # Copy logs for LipSync service only"
    echo "  $0 asd          # Copy logs for ASD service only"
    echo "  $0 controller   # Copy logs for Controller service only"
    echo ""
    echo "Log files will be saved to:"
    echo "  ./logs/s2s.log"
    echo "  ./logs/lipsync.log"
    echo "  ./logs/asd.log"
    echo "  ./logs/controller.log"
}

# Main script logic
main() {
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running or not accessible"
        exit 1
    fi
    
    # If no arguments provided, copy all logs
    if [ $# -eq 0 ]; then
        copy_all_logs
        exit 0
    fi
    
    # If help is requested
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        show_usage
        exit 0
    fi
    
    # Handle specific service
    case "$1" in
        "s2s")
            copy_service_logs "s2s" "s2s"
            ;;
        "ast")
            copy_service_logs "ast" "ast"
            ;;
        "tts")
            copy_service_logs "tts" "tts"
            ;;
        "lipsync")
            copy_service_logs "lipsync" "lipsync"
            ;;
        "asd")
            copy_service_logs "asd" "asd"
            ;;
        "controller")
            copy_service_logs "controller" "controller"
            ;;
        *)
            print_error "Unknown service: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 