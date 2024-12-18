from sphinx.ext.autodoc import ClassDocumenter, MethodDocumenter, FunctionDocumenter

def setup(app):
    app.add_autodocumenter(RawFunctionDocumenter)
    app.add_autodocumenter(RawMethodDocumenter)
    app.add_autodocumenter(RawClassDocumenter)
    return {'version': '1.0', 'parallel_read_safe': True}

class RawDocumenterMixin:
    def add_directive_header(self, sig):
        super().add_directive_header(sig)
        if self.object.__doc__:
            # Add raw docstring as a code block
            self.add_line('', '<autodoc>')
            self.add_line('.. code-block:: python', '<autodoc>')
            self.add_line('', '<autodoc>')
            for line in self.object.__doc__.splitlines():
                self.add_line(f'    {line}', '<autodoc>')
            self.add_line('', '<autodoc>')

class RawFunctionDocumenter(RawDocumenterMixin, FunctionDocumenter):
    objtype = 'function'

class RawMethodDocumenter(RawDocumenterMixin, MethodDocumenter):
    objtype = 'method'

class RawClassDocumenter(RawDocumenterMixin, ClassDocumenter):
    objtype = 'class'
