useless_column = ['remote_ip', 'ident', 'user', 'time', 'timezone', 'size', 'referrer']
keywords_url = ('%27%20OR%201=1--%27', 'OR+1=1', 'OR 1=1')
keywords_agent = ('gobuster', 'nikto', 'shell', 'nmap', 'curl')
keywords_response = ('200', '302')
keywords_request = ('GET', 'POST', 'PROFIND', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'TRACE', 'CONNECT')
parser_header = ['remote_ip', 'ident', 'user', 'time', 'timezone', 'url', 'response_code', 'size', 'referrer', 'useragent']