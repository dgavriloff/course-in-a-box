# Setup and How to Run Guide

This guide will help you set up and run Course-in-a-Box on your local machine for development and testing.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start with Docker (Recommended)](#quick-start-with-docker-recommended)
- [Local Installation without Docker](#local-installation-without-docker)
- [Running the Development Server](#running-the-development-server)
- [Making Changes](#making-changes)
- [Building for Production](#building-for-production)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before you begin, ensure you have one of the following installed:

### Option 1: Docker (Recommended)
- [Docker Desktop](https://docs.docker.com/get-docker/) or Docker Engine
- Docker Compose (usually included with Docker Desktop)

### Option 2: Local Ruby Installation
- Ruby 2.7 or higher (Ruby 3.x recommended)
- Bundler gem (`gem install bundler`)
- Git

## Quick Start with Docker (Recommended)

Docker is the easiest way to get started as it handles all dependencies automatically.

### Using Docker Compose

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/your-username/course-in-a-box.git
   cd course-in-a-box
   ```

2. **Start the development server**:
   ```bash
   docker compose up
   ```
   
   Note: Older Docker versions may require `docker-compose` (with hyphen) instead.

3. **View your course**:
   Open your browser and navigate to `http://localhost:4000`

4. **Stop the server**:
   Press `Ctrl+C` in the terminal, or run:
   ```bash
   docker compose down
   ```

### Using Docker Run Command

Alternatively, you can run Docker directly without Docker Compose:

```bash
docker run -i -t --rm -u 1000:1000 -p 4000:4000 \
  -v "$(pwd)":/opt/app \
  -v "$(pwd)/.bundler":/opt/bundler \
  -e BUNDLE_PATH=/opt/bundler \
  -w /opt/app \
  ruby:3.3 \
  bash -c "bundle install && bundle exec jekyll serve --watch -H 0.0.0.0"
```

**Note for Windows users**: Replace `$(pwd)` with `%cd%` in Command Prompt or `${PWD}` in PowerShell.

## Local Installation without Docker

If you prefer to run Jekyll natively on your machine:

### 1. Install Ruby

#### macOS
Ruby comes pre-installed on macOS, but you might want to use a version manager like rbenv:
```bash
brew install rbenv
rbenv install 3.3.0
rbenv global 3.3.0
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install ruby-full build-essential zlib1g-dev
```

#### Windows
Download and install from [RubyInstaller](https://rubyinstaller.org/)

### 2. Install Bundler

```bash
gem install bundler
```

### 3. Install Dependencies

From the project root directory:
```bash
bundle install
```

### 4. Run the Development Server

```bash
bundle exec jekyll serve --watch
```

Your site will be available at `http://localhost:4000`

## Running the Development Server

Once you have either Docker or local Ruby/Jekyll set up:

### Development Mode (with auto-reload)
The development server watches for file changes and automatically rebuilds:
- **Docker Compose**: `docker compose up` (or `docker-compose up` for older versions)
- **Local**: `bundle exec jekyll serve --watch`

### Production-like Build
To test the production build locally:
```bash
bundle exec jekyll serve --no-watch
```

### Custom Port
To run on a different port (e.g., 4001):
```bash
bundle exec jekyll serve --port 4001
```

## Making Changes

### File Structure
- `_config.yml` - Main configuration file
- `_layouts/` - HTML templates
- `_includes/` - Reusable HTML snippets
- `modules/` - Course content
- `css/` - Stylesheets
- `js/` - JavaScript files
- `img/` - Images

### Editing Content

1. **Course modules** are stored in `modules/` directory
2. **Homepage** can be edited in `index.md`
3. **Styles** are in `css/` and `_sass/` directories
4. **Configuration** is in `_config.yml`

### Viewing Changes

With `--watch` flag enabled, changes to files will automatically trigger a rebuild. Simply refresh your browser to see the changes (it may take a few seconds to rebuild).

**Note**: Changes to `_config.yml` require restarting the Jekyll server.

## Building for Production

To build the static site for deployment:

```bash
bundle exec jekyll build
```

This generates the production-ready site in the `_site/` directory.

## Troubleshooting

### Port Already in Use

If port 4000 is already in use:
```bash
# Find the process using port 4000
lsof -i :4000  # macOS/Linux
netstat -ano | findstr :4000  # Windows

# Run on a different port
bundle exec jekyll serve --port 4001
```

### Permission Errors with Docker

If you encounter permission errors with Docker, ensure the user ID matches:
```bash
docker compose run --rm -u $(id -u):$(id -g) jekyll bash -c "bundle install && bundle exec jekyll serve -H 0.0.0.0"
```

### Bundle Install Fails

Try clearing the bundle cache:
```bash
rm -rf .bundler/
rm Gemfile.lock
bundle install
```

### Jekyll Not Found

Make sure you've run `bundle install` and use `bundle exec` before Jekyll commands:
```bash
bundle exec jekyll serve
```

### Changes Not Appearing

1. **Hard refresh** your browser (Ctrl+Shift+R or Cmd+Shift+R)
2. **Check the terminal** for build errors
3. **Restart Jekyll** if you changed `_config.yml`
4. **Clear Jekyll cache**: `bundle exec jekyll clean`

### Docker Build Issues

If Docker builds are failing:
```bash
# Remove old containers and images
docker compose down
docker system prune

# Rebuild from scratch
docker compose up --build
```

Note: Older Docker versions use `docker-compose` (with hyphen) instead of `docker compose`.

## Additional Resources

- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [Course-in-a-Box Documentation](https://course-in-a-box.p2pu.org)
- [P2PU Community Forum](https://community.p2pu.org/c/tech/course-in-a-box/78)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [P2PU Community Forum](https://community.p2pu.org/c/tech/course-in-a-box/78)
2. Search existing [GitHub Issues](https://github.com/p2pu/course-in-a-box/issues)
3. Create a new issue with details about your problem

## Next Steps

After setting up your local development environment:

1. Explore the [Course-in-a-Box documentation](https://course-in-a-box.p2pu.org)
2. Customize your course by editing files in the `modules/` directory
3. Adjust the theme and styling in `_config.yml` and `css/` directory
4. Add your content and publish to GitHub Pages

---

Happy course building! 🎓
