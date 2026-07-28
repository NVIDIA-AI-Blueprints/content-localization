# Content Localization Blueprint Documentation

This directory contains comprehensive documentation for the Content Localization Blueprint, a solution for translating and dubbing audio and video content using AI services.

## Documentation Structure

### Core Documentation

- **`architecture.rst`** - System architecture overview and component relationships
- **`client_types.rst`** - Comprehensive guide to Controller, Direct, and Individual clients

### Existing Documentation

- **`setup.rst`** - Setup and installation instructions
- **`client_quickstart.rst`** - Quick start guide for clients
- **`client.rst`** - Client API documentation
- **`client_troubleshooting.rst`** - Troubleshooting guide
- **`logging.rst`** - Logging configuration and debugging

## Building Documentation

### Prerequisites

- Python 3.12+
- Virtual environment with development dependencies installed
- Node.js/npm (optional, only for UI development and pre-commit hooks)

### Setup

Before building the documentation, install the required dependencies:

```bash
# Install Python development dependencies (includes Sphinx and related tools)
uv sync --extra docs
```

Install Node.js from: https://nodejs.org/en/download

This will install:
- `sphinx` - Documentation generator
- `sphinx-rtd-theme` - Read the Docs theme
- `recommonmark` - Markdown support for Sphinx
- Other development tools (ruff, pre-commit)

**Note**: The Node.js/npm setup is only required if you're working on the `client/demos` project or running pre-commit hooks with UI formatting/linting.

### Build Commands

```bash
# Build HTML documentation (from the repo root)
bash docs/build_docs.sh

# Build specific format
sphinx-build -b html docs/source build/docs/html
sphinx-build -b epub docs/source build/docs/epub
```

### Output

The built documentation will be available in:

- **HTML**: `build/docs/html/index.html`
- **EPUB**: `build/docs/epub/index.epub`

## Documentation Features

### Architecture Documentation

- **System Overview**: Complete system architecture and component relationships
- **Controller Pipeline**: Detailed explanation of push-mode orchestration
- **Client Types**: Comprehensive guide to all client architectures
- **Performance Characteristics**: Memory, CPU, latency, and throughput analysis

### Visual Documentation

- **Mermaid Diagrams**: In-page Mermaid diagrams for architecture and workflow visuals
- **Flow Diagrams**: Sequence and data flow diagrams
- **Comparison Charts**: Side-by-side comparisons of modes and clients

### Code Examples

- **Configuration Examples**: Service and client configuration
- **Usage Examples**: Practical code examples for all client types
- **Best Practices**: Recommended practices for different scenarios

## Key Topics Covered

### Controller Processing Mode

1. **Push Mode**
   - Multi-threaded processing for real-time streaming
   - Background threads for non-blocking operation
   - Suitable for streaming applications

### Client Types

1. **Controller Client**
   - Single gRPC connection to Controller service
   - Simplified orchestration and easy integration
   - Production-ready with minimal configuration

2. **Direct Client**
   - Multiple gRPC connections to individual services
   - Full control over service interactions
   - Custom orchestration and performance optimization

3. **Individual Clients**
   - Service-specific clients for S2S, ASD, and LipSync
   - Focused functionality with low complexity
   - Easy testing and debugging of individual services

### AI Services

1. **Speech-to-Speech (S2S) Service**
   - Audio translation and synthesis
   - ElevenLabs or CambAI backend support
   - Real-time streaming capabilities

2. **Active Speaker Detection (ASD) NIM Service**
   - Speaker identification in video content
   - Speaker info generation with confidence scoring
   - GPU/CPU fallback support

3. **LipSync Service**
   - Lip synchronization with translated audio
   - Advanced encoding options
   - Multiple output formats

## Documentation Maintenance

### Adding New Content

1. Create new `.rst` files in `docs/source/`
2. Update `docs/source/index.rst` to include new content
3. Rebuild documentation using `bash docs/build_docs.sh` (from the repo root)

### Updating Mermaid Diagrams

1. Update Mermaid blocks directly in docs pages under `docs/source/`
2. Keep reusable Mermaid source files in `docs/source/uml_mermaid/`
3. Rebuild documentation using `bash docs/build_docs.sh` (from the repo root)
4. Verify rendered diagrams and cross-references

### Style Guidelines

- Use reStructuredText (RST) format for documentation
- Include code examples with syntax highlighting
- Add cross-references between related topics
- Maintain consistent formatting and structure

## Troubleshooting

### Common Issues

- **Build Errors**: Check Python dependencies and Sphinx installation
- **Missing Diagrams**: Validate Mermaid syntax and ensure docs pages contain valid `.. mermaid::` directives
- **Formatting Issues**: Validate RST syntax and structure

### Getting Help

- Check existing documentation for common solutions
- Review troubleshooting guides in `client_troubleshooting.rst`
- Consult logging documentation for debugging information

## Contributing

When contributing to documentation:

1. Follow existing style and structure
2. Include code examples and practical usage
3. Update related documentation as needed
4. Test documentation builds successfully
5. Validate all links and references
