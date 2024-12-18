from sphinx.ext.autodoc import between

def setup(app):
    # Register a priority handler for docstrings
    app.connect('autodoc-process-docstring', process_docstring)
    return {'version': '1.0', 'parallel_read_safe': True}

def process_docstring(app, what, name, obj, options, lines):
    """Process docstrings to ensure they are properly formatted."""
    # Remove empty lines at start
    while lines and not lines[0].strip():
        lines.pop(0)
    # Remove empty lines at end
    while lines and not lines[-1].strip():
        lines.pop()
    
    # Ensure there's spacing between sections
    new_lines = []
    in_section = False
    for line in lines:
        if line.strip() and line[0].isalpha() and ':' in line:
            if in_section:
                new_lines.append('')
            in_section = True
        new_lines.append(line)
    
    lines[:] = new_lines
