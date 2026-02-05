# Contributing to Paddle-Iluvatar

Thank you for your interest in contributing to Paddle-Iluvatar! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Paddle-Iluvatar.git
   cd Paddle-Iluvatar
   ```
3. Set up the development environment:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure they follow the coding standards

3. Add tests for your changes:
   ```bash
   # Add tests to tests/ directory
   python -m pytest tests/test_your_feature.py
   ```

4. Run all tests to ensure nothing is broken:
   ```bash
   python -m pytest tests/ -v
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

6. Push to your fork and submit a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Standards

### Python Code

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Format code with `black`:
  ```bash
  black python/paddle_iluvatar
  ```
- Sort imports with `isort`:
  ```bash
  isort python/paddle_iluvatar
  ```
- Check code with `flake8`:
  ```bash
  flake8 python/paddle_iluvatar
  ```

### C++ Code

- Follow Google C++ Style Guide
- Use C++14 features
- Add comments for complex logic
- Use RAII for resource management
- Format code consistently

### Documentation

- Update README.md if adding new features
- Add API documentation for new functions
- Update docs/ if changing architecture
- Include examples for new features

## Testing Guidelines

### Unit Tests

- Write tests for all new functionality
- Maintain or improve test coverage
- Use descriptive test names
- Test both success and failure cases

### Example Test Structure

```python
import unittest
import paddle_iluvatar.device as device

class TestNewFeature(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        if not device.is_available():
            self.skipTest("No GPU available")
    
    def test_feature_success(self):
        """Test successful operation"""
        result = your_function()
        self.assertEqual(result, expected_value)
    
    def test_feature_failure(self):
        """Test error handling"""
        with self.assertRaises(ValueError):
            your_function(invalid_input)
```

## Pull Request Process

1. **Description**: Provide a clear description of what your PR does
2. **Testing**: Ensure all tests pass
3. **Documentation**: Update documentation as needed
4. **Review**: Be responsive to review feedback
5. **Merge**: Once approved, your PR will be merged

### PR Checklist

- [ ] Code follows project coding standards
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts with main branch

## Reporting Bugs

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal steps to reproduce the issue
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: Python version, OS, GPU model, etc.
6. **Logs**: Relevant error messages or logs

### Bug Report Template

```markdown
**Description**
Brief description of the bug

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- Python version:
- OS:
- Paddle-Iluvatar version:
- GPU model:

**Additional Context**
Any other relevant information
```

## Requesting Features

When requesting features, please include:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Additional Context**: Any other relevant information

## Project Structure

```
paddle-iluvatar/
├── .github/          # GitHub Actions workflows
├── csrc/             # C++ implementation
├── docs/             # Documentation
├── examples/         # Usage examples
├── include/          # C++ headers
├── python/           # Python bindings
│   └── paddle_iluvatar/
│       ├── __init__.py
│       ├── device.py
│       ├── memory.py
│       └── stream.py
├── tests/            # Unit tests
├── CMakeLists.txt    # CMake build
├── setup.py          # Python package setup
└── README.md         # Main documentation
```

## Areas for Contribution

### High Priority

1. **SDK Integration**: Replace stub implementations with actual Iluvatar SDK
2. **Operator Kernels**: Implement GPU kernels for common operations
3. **PaddlePaddle Integration**: Register as custom device backend
4. **Performance Optimization**: Optimize memory management and execution

### Medium Priority

1. **Documentation**: Improve guides and tutorials
2. **Examples**: Add more usage examples
3. **Testing**: Increase test coverage
4. **Benchmarks**: Add performance benchmarks

### Low Priority

1. **Code Quality**: Refactoring and cleanup
2. **CI/CD**: Improve build and test pipelines
3. **Tooling**: Development tools and utilities

## Communication

- **GitHub Issues**: For bugs and feature requests
- **Pull Requests**: For code contributions
- **Discussions**: For questions and general discussion

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Maintain a positive environment

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search existing issues
3. Open a new issue with your question

Thank you for contributing to Paddle-Iluvatar!
