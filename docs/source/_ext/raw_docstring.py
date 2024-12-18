from sphinx.ext.autodoc import ClassDocumenter, MethodDocumenter, FunctionDocumenter
import inspect

class RawDocumenterMixin:
    def get_doc(self, encoding=None, ignore=1):
        """Directly return the docstring as-is."""
        if self.object.__doc__:
            # Return as a list of lines
            return [self.object.__doc__.splitlines()]
        return []

    def add_content(self, more_content):
        """Add content from docstrings, attribute documentation and user."""
        docstring = self.get_doc()
        if docstring:
            self.add_line('', '<autodoc>')
            self.add_line('**Raw Docstring:**', '<autodoc>')
            self.add_line('', '<autodoc>')
            self.add_line('.. code-block:: python', '<autodoc>')
            self.add_line('', '<autodoc>')
            for line in docstring[0]:
                self.add_line(f'    {line}', '<autodoc>')
            self.add_line('', '<autodoc>')
        
        # Add any additional content from parent class
        super().add_content(more_content)

class RawFunctionDocumenter(RawDocumenterMixin, FunctionDocumenter):
    objtype = 'function'
    priority = 10 + FunctionDocumenter.priority

class RawMethodDocumenter(RawDocumenterMixin, MethodDocumenter):
    objtype = 'method'
    priority = 10 + MethodDocumenter.priority

class RawClassDocumenter(RawDocumenterMixin, ClassDocumenter):
    objtype = 'class'
    priority = 10 + ClassDocumenter.priority

def setup(app):
    app.add_autodocumenter(RawFunctionDocumenter)
    app.add_autodocumenter(RawMethodDocumenter)
    app.add_autodocumenter(RawClassDocumenter)
    return {'version': '1.0', 'parallel_read_safe': True}
