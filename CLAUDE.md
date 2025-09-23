# Claude Code Rules and Guidelines

This file contains important rules and guidelines that Claude must follow when working on this project.

## Code Quality Rules

### **Pre-Completion Checklist**
**RULE**: Always run these commands before declaring any work complete:
1. `pdm format` - Format code
2. `pdm lint` - Run linting checks
3. `pdm test` - Run all tests

**Never forget this rule.** All three commands must pass successfully before work is considered done.

### **Unicode Policy**
**RULE**: Never use Unicode characters (emojis, special symbols, etc.) in any code, output, or files.
- Use plain ASCII text only
- Replace emojis with text descriptions (e.g., "WARNING:" instead of warning symbols)
- This prevents encoding issues on Windows systems

## Project Guidelines

- Follow existing code conventions and patterns
- Use shared functions when multiple nodes need the same functionality. 
- Keep code DRY (Don't Repeat Yourself)
- Write comprehensive tests for new functionality
- Update documentation when adding new features

---
*This file will be updated as new rules and guidelines are established.*