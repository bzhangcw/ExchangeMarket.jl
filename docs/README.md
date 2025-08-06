# ExchangeMarket.jl Documentation

This directory contains the documentation for ExchangeMarket.jl, built using [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl).

## Building the Documentation

### Prerequisites

- Julia 1.6 or later
- ExchangeMarket.jl package installed

### Local Development

1. Navigate to the docs directory:
   ```bash
   cd docs
   ```

2. Install documentation dependencies:
   ```julia
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

3. Build the documentation:
   ```julia
   julia --project=. docs/make.jl
   ```

4. View the documentation:
   - Open `docs/build/index.html` in your web browser
   - Or serve locally using a web server

### Continuous Integration

The documentation is automatically built and deployed via GitHub Actions when:
- Changes are pushed to the `main` branch
- A new version tag is created
- A pull request is opened

## Documentation Structure

- `src/`: Source files for the documentation
  - `index.md`: Main landing page
  - `getting_started.md`: Installation and basic usage guide
  - `api/`: API reference documentation
  - `examples/`: Code examples
  - `tutorials/`: Step-by-step tutorials
- `make.jl`: Documenter.jl build script
- `Project.toml`: Documentation dependencies

## Contributing to Documentation

1. Edit the Markdown files in `src/`
2. Add docstrings to your Julia code using `@doc` or `@docs`
3. Test locally: `julia --project=. docs/make.jl`
4. Submit a pull request

## Documentation Features

- **API Reference**: Automatically generated from docstrings
- **Examples**: Complete, runnable code examples
- **Tutorials**: Step-by-step guides for common tasks
- **Search**: Full-text search across all documentation
- **Versioning**: Documentation for different package versions

## Deployment

The documentation is deployed to GitHub Pages at:
`https://yourusername.github.io/ExchangeMarket.jl/`

To set up deployment:

1. Create a GitHub repository
2. Add the `DOCUMENTER_KEY` secret to your repository
3. Push to the `main` branch to trigger automatic deployment 