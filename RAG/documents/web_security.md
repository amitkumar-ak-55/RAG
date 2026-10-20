# Web Security

Input validation should be performed on the server using allowlists and type checks appropriate to each field. Output encoding must match the output context, such as HTML, JavaScript, URL, or SQL, to reduce injection risk.

Web applications should use secure session cookies, HTTPS, CSRF protections where applicable, and security headers. Authentication and authorization must be enforced on every sensitive server-side action rather than only in the user interface.