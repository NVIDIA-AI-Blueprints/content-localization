# Functional Tests

This directory contains comprehensive end-to-end functional tests for all Content-Localization clients. These tests validate the complete pipeline by running actual clients with real inputs and verifying outputs.

## Overview

The functional tests are designed to:

- **Run actual clients** with sample inputs
- **Validate outputs** for correct format, size, and content
- **Test different configurations** (chunk sizes, quality settings, etc.)
- **Verify service communication** between clients and services
- **Ensure end-to-end functionality** of the complete pipeline

## Test Structure

### Individual Client Tests

Each client has its own comprehensive test file with pytest-compatible test functions:

- **`test_controller_client.py`** - Tests the Controller client (orchestrates all services)
- **`test_direct_client.py`** - Tests the Direct client (direct service communication)
- **`test_s2s_client.py`** - Tests the S2S client (audio translation)
- **`test_lipsync_client.py`** - Tests the LipSync client (lip-sync processing)
- **`test_asd_client.py`** - Tests the ASD client (speaker detection)

### Test Configuration

- **`pytest.ini`** - Pytest configuration for functional tests

### Test Functions

Each test file contains comprehensive test functions that pytest automatically discovers:

- `test_*_service_health()` - Check if required services are running
- `test_input_files_exist()` - Verify input files are available
- `test_*_client_comprehensive()` - Complete functionality tests covering basic and complex features

### Chunk Size Configuration

The tests use optimized chunk sizes for better streaming performance:

- **Video chunks**: 1MB (1048576 bytes) for basic tests, 512KB (524288 bytes) for custom tests
- **Audio chunks**: 2.0 seconds for basic tests, 1.0 seconds for custom tests

## Prerequisites

Before running the functional tests, ensure:

1. **All services are running:**
   - S2S Service: `localhost:50050`
   - LipSync Service: `localhost:50054`
   - ASD NIM Service: `localhost:50055`
   - Controller Service: `localhost:50056`

2. **Input files exist:**
   - `assets/sample_audio.wav`
   - `assets/sample_video_streamable.mp4`

3. **Python environment is set up** with all required dependencies

## Running Tests

### Run All Tests with Pytest

To run all functional tests using pytest:

```bash
# From project root
python -m pytest functional_tests/ -v

# Or from functional_tests directory
cd functional_tests
python -m pytest -v
```

This will:

- Automatically discover all test functions
- Run tests with detailed reporting
- Provide comprehensive pass/fail results

### Run Individual Test Files

To run a specific client test:

```bash
# From project root
python -m pytest functional_tests/test_s2s_client.py -v
python -m pytest functional_tests/test_asd_client.py -v
python -m pytest functional_tests/test_lipsync_client.py -v
python -m pytest functional_tests/test_direct_client.py -v
python -m pytest functional_tests/test_controller_client.py -v

# Or from functional_tests directory
cd functional_tests
python -m pytest test_s2s_client.py -v
python -m pytest test_asd_client.py -v
python -m pytest test_lipsync_client.py -v
python -m pytest test_direct_client.py -v
python -m pytest test_controller_client.py -v
```

### Run with Additional Options

```bash
# Verbose output with short tracebacks
python -m pytest functional_tests/ -v --tb=short

# Run with color output
python -m pytest functional_tests/ -v --color=yes

# Run specific test functions
python -m pytest functional_tests/ -k "test_s2s_client_basic" -v
```

### Selecting Source and Target Languages

The tests fall back to each client's built-in defaults, which use
ElevenLabs-style language **codes** (e.g. `en`, `es`). A **CambAI** deployed
stack expects integer language **IDs** instead, so running with the defaults
against CambAI fails with an error such as:

```text
Invalid CambAI source language ID: en. Must be a valid CambAI integer ID string.
```

When testing against a CambAI stack, pass the source and target languages
explicitly as CambAI integer IDs (e.g. `1` = English, `54` = Spanish):

```bash
source .venv/bin/activate && python -m pytest functional_tests/ -v --require-services \
    --source-language 1 --target-language 54
```

The same values may instead be provided via the `TEST_SOURCE_LANGUAGE` and
`TEST_TARGET_LANGUAGE` environment variables. See the main
[README](../README.md) CambAI section for the full language ID mapping.

## Test Coverage

Each client test includes:

### Controller Client Tests

- Service health check
- Input file validation
- Comprehensive functionality test (basic + complex features)

### Direct Client Tests

- Service health check (S2S, LipSync, ASD)
- Input file validation
- Comprehensive functionality test (basic + complex features)

### S2S Client Tests

- Service health check
- Input file validation
- Comprehensive functionality test (basic + complex features + latency analysis)

### LipSync Client Tests

- Service health check
- Input file validation
- Comprehensive functionality test (basic + complex features)

### ASD Client Tests

- Service health check
- Input file validation
- Comprehensive functionality test (basic + complex features)

## Output Validation

Each test validates:

### Video Outputs (MP4)

- File exists and is not empty
- Valid MP4 format (checks file headers)
- Reasonable file size

### Audio Outputs (MP3)

- File exists and is not empty
- Valid MP3 format (checks ID3/MPEG headers)
- Reasonable file size

### CSV Outputs (ASD)

- File exists and is not empty
- Valid CSV format
- Proper data structure

## Test Configuration

### Timeouts

- Individual client tests: 5 minutes
- Complete test suite: 10 minutes per test
- Service health checks: 30 seconds

### Output File Preservation

All test outputs are preserved in the `functional_tests/outputs/` directory for later inspection:

#### Output File Naming Convention

- **Controller tests**: `controller_comprehensive_output.mp4`
- **Direct tests**: `direct_comprehensive_output.mp4`, `direct_comprehensive_audio.mp3`
- **S2S tests**: `s2s_comprehensive_output.mp3`, `s2s_comprehensive_latency_plot.png`
- **LipSync tests**: `lipsync_comprehensive_output.mp4`
- **ASD tests**: `asd_comprehensive_output.csv`

#### Cleanup Process

- **Before each test run**: All previous outputs are automatically cleaned up
- **After each test**: Output files are preserved for inspection
- **Manual cleanup**: You can manually delete files in `functional_tests/outputs/`

#### Example Output Files

```
functional_tests/outputs/
├── controller_comprehensive_output.mp4
├── direct_comprehensive_output.mp4
├── direct_comprehensive_audio.mp3
├── s2s_comprehensive_output.mp3
├── s2s_comprehensive_latency_plot.png
├── lipsync_comprehensive_output.mp4
└── asd_comprehensive_output.csv
```

### Error Handling

- Graceful handling of service unavailability
- Detailed error reporting for debugging
- Timeout protection for hanging processes

## Expected Results

### Successful Test Run

```
ALL FUNCTIONAL TESTS PASSED!
All clients are working correctly
All services are communicating properly
All outputs are being generated successfully
```

### Failed Test Run

```
2 TEST SUITE(S) FAILED!
Please check the logs above for specific failure details.
```

## Troubleshooting

### Common Issues

1. **Services not running**
   - Start all required services before running tests
   - Check service ports and connectivity

2. **Input files missing**
   - Ensure sample files exist in `assets/`
   - Check file permissions

3. **Timeout errors**
   - Increase system resources
   - Check for network issues
   - Verify service performance

4. **Format validation failures**
   - Check service configurations
   - Verify input file formats
   - Review service logs

### Debug Mode

For detailed debugging, you can run individual tests and examine:

- STDOUT/STDERR output
- Temporary output files
- Service logs
- Network connectivity

## Performance Considerations

- Tests use temporary directories to avoid disk space issues
- Timeouts prevent infinite hangs
- Parallel execution not recommended due to service dependencies
- Consider running on dedicated test environment
