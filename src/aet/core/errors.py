class AetError(Exception): pass
class SuiteNotFoundError(AetError): pass
class RunAlreadyExistsError(AetError): pass
class ValidationError(AetError): pass
class ExecutionError(AetError): pass
class TemplateError(AetError): pass
