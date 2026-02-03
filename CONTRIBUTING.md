# Contributing to Grand Lyon Photo Clusters

Thank you for your interest in contributing to this project! 🎉

## How to Contribute

### Reporting Issues

- Before opening an issue, please check if a similar issue already exists
- Use the issue templates when available
- Provide as much context as possible (OS, Python version, error messages)

### Submitting Pull Requests

1. **Fork the repository** and create a new branch from `main`
2. **Install dependencies**: 
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Make your changes** with clear, descriptive commits
4. **Test your changes** by running the pipeline:
   ```bash
   python scripts/run_full_pipeline.py
   ```
5. **Update documentation** if needed
6. **Submit a Pull Request** with a clear description of changes

### Code Style

- Follow [PEP 8](https://pep8.org/) for Python code
- Use meaningful variable and function names
- Add docstrings to new functions and classes
- Keep functions focused and modular

### Areas for Contribution

- 🐛 **Bug fixes** - Found a bug? Feel free to fix it!
- 📖 **Documentation** - Improve README, add examples, fix typos
- ✨ **New features** - New clustering algorithms, visualization improvements
- 🧪 **Tests** - Add unit tests for existing functionality
- 🌍 **Translations** - Translate documentation to other languages

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/grandlyon-photo-clusters.git
cd grandlyon-photo-clusters

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python scripts/run_full_pipeline.py
```

## Project Structure

```
src/           # Core Python modules
scripts/       # Pipeline and utility scripts
notebooks/     # Jupyter notebooks for experimentation
data/          # Dataset files (not tracked in git)
reports/       # Generated analysis reports
app/           # Interactive map outputs
```

## Questions?

If you have questions, feel free to open an issue with the "question" label.

---

Thank you for contributing! 🙏
