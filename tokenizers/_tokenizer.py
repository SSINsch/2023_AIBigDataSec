import re
import logging
import html
import urllib

logger = logging.getLogger(__name__)


class LogTokenizer:
    def __init__(self, postprocessing=False):
        self.postprocessing = postprocessing

    def tokenize(self, sentence):
        # tokenize string
        
        def preprocess(payload):
            typo = {"host": "Host", "content-length": "Content-Length", "User-agent": "User-Agent"}
            payload = html.unescape(payload)
            payload = urllib.parse.unquote(payload)
            payloads = {}
                
            if "\\\\r\\\\n" in payload:
                payload = payload.replace("\\\\", "\\")

            if " HTTP/1.1\\r\\n" in payload:
                data = payload.split(" HTTP/1.1\\r\\n")[0]
                headers = payload.split(" HTTP/1.1\\r\\n")[1]
                headers = headers.split("\\r\\n")
                #payloads["HTTP_Version"] = 1.1
            elif " HTTP/1.0\\r\\n" in payload:
                data = payload.split(" HTTP/1.0\\r\\n")[0]
                headers = payload.split(" HTTP/1.0\\r\\n")[1]
                headers = headers.split("\\r\\n")
                #payloads["HTTP_Version"] = 1.0
            elif " HTTP/1.1\\n" in payload:
                data = payload.split(" HTTP/1.1\\n")[0]
                headers = payload.split(" HTTP/1.1\\n")[1]
                headers = headers.split("\\n")
                #payloads["HTTP_Version"] = 1.1
            elif " HTTP/1.0\\n" in payload:
                data = payload.split(" HTTP/1.0\\n")[0]
                headers = payload.split(" HTTP/1.0\\n")[1]
                headers = headers.split("\\n")
                #payloads["HTTP_Version"] = 1.0
                
            payloads["Method"] = data.split(" ", maxsplit=1)[0]
            url = data.split(" ", maxsplit=1)[1]
            if "?" in url:
                payloads["Path"] = url.split("?", maxsplit=1)[0]
                payloads["Parameter"] = url.split("?", maxsplit=1)[1][:-1]
                
            else:
                payloads["Path"] = url
            
            for j in range(len(headers)):
                header = headers[j]
                if header == "":
                    body = ''.join(headers[j + 1:])
                    payloads["Body"] = body
                    break
                if ":" in header:
                    a = header.split(":", maxsplit=1)[0]
                    if a in typo:
                        a = typo[a]
                    b = header.split(":", maxsplit=1)[1][1:]
                    payloads[a] = b
        
            return payloads
        
        def preprocess2(payload):
            for header in payload:
                if header in ["Payload", "Label"]:
                    continue
                data = payload[header]
                data = urllib.parse.unquote(data)
                
                if header == "User-Agent":
                    #data = re.sub(r'sqlmap/.*? ', 'sqlmap ', data)
                    # Mozilla/5.00 -> Mozilla
                    data = re.sub(r'/v?\d+(\.\d+)*', '', data)
                    # Test:003527 -> Test
                    data = re.sub(r'Test:.*\)', 'Test)', data)
                    # rv:92.0 삭제
                    data = re.sub(r'rv:\d+(\.\d+)*', '', data)
                    # like Gecko 삭제
                    data = re.sub(r'KHTML, like Gecko', 'KHTML', data)
                    
                elif header == "Content-Type":
                    # ----WebKitFormBoundaryj2RARupKJMmCihRv -> WebKitFormBoundary
                    data = re.sub(r'----WebKitFormBoundary.*', 'WebKitFormBoundary', data)
                    
                elif header == "Cookie":
                    # a:15:{i:0;i:38;i:1;i:47;i:2;i:41;i:3;i:56;i:4;i:51;i:5;i:42;i:6;i:79;i:7;i:43;i:8;i:77;i:9;i:75;i:10;i:68;i:11;i:54;i:12;i:70;i:13;i:55;i:14;i:52;} -> serialize
                    data = re.sub(r'a:\d+:{.*?}', 'serialize', data)
                    # designart_site=cg0q4hq5j1rgo9o2ns7s3nsnjlauq1rp 랜덤값 삭제
                    data = re.sub(r'=[a-z\d]{25,32}', '', data)
                    
                elif header == "Accept":
                    # */* -> all
                    data = re.sub(r'\*/\*', 'all', data)
                elif header == "Accept-Encoding":
                    # q=0.9 삭제
                    data = re.sub(r'q=\d+(\.\d+)?', '', data)
                elif header == "Accept-Language":
                    data = re.sub(r'q=\d+(\.\d+)?', '', data)
                
                data = re.sub(r'(CHR\(\d+\)(\|\|CHR\(\d+\))*)', 'CHR', data)
                data = re.sub(r'(CHR(\+CHR)*)', 'CHR', data)
                data = re.sub(r'(CHAR\(\d+\)(\|\|CHR\(\d+\))*)', 'CHAR', data)
                data = re.sub(r'(CHAR(\+CHAR)*)', 'CHR', data)
                
                data = re.sub(r'q=\d+(\.\d+)?', '', data)
                    
                data = re.sub(r'<!--|-->', ' ', data)
                data = re.sub(r"\r|\n|!%", ' ', data)
                data = re.sub(r'-CVE-\d{4}-\d{4}', ' CVE', data)
                
                data = re.sub(r'[,?{}()<>&@=;:/+\\\[\]\|]', ' ', data)
                data = re.sub(r'[`\"\']', '', data)
                data = re.sub(r'--', ' --', data)
                
                if header == "Path" or header == "Referer":
                    # phpMyAdmin-2.5.5-rc2 -> phpMyAdmin rc2
                    data = re.sub(r'-(\d+(\.\d+)*)-', ' ', data)
                    data = re.sub(r'-(\d+(\.\d+)*)', '', data)
                    # sub_04_1_read.php -> sub_read.php
                    data = re.sub(r'_(\d+(_\d+)*)', '', data)
                    # /로 끝날 경우 DIREC 추가
                    if payload[header] == "" or payload[header][-1] == "/":
                        data += "DIREC"
                
                elif header == "User-Agent":
                    data = data.replace("Windows NT 10.0", "Windows10")
                    data = data.replace("Windows NT 5.1", "WindowsXP")
                    data = data.replace("Windows NT 6.1", "Windows7")
                    data = data.replace("Windows NT 6.0", "WindowsVista")
                    data = re.sub(r'Mac OS X [\d_\.]+', 'MacOS', data)
                    data = re.sub(r'MSIE [\d.]+', 'MSIE', data)

                elif header == "Host":
                    data = re.sub(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})([a-zA-Z])', r'\1 \2', data)

                elif header == "Content-Type":
                    data = re.sub(r'[%.]', ' ', data)
                    data = re.sub(r'[#_]', '', data)

                elif header == "Method":
                    data = re.sub(r'\\u[0-9]{4}', ' ', data)
                    data = re.sub(r'\\x[0-9]{2}', ' ', data)

                elif header == "Authorization":
                    if "Negotiate" in data:
                        data = "Negotiate"
                    elif "Basic" in data:
                        data = "Basic"

                elif header == "X-Arachni-Scan-Seed":
                    data = ""
                
                pattern = r'\b(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.' \
                    r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.' \
                    r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.' \
                    r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
                data = re.sub(pattern, "ip", data)
                pattern = r'www\.\S+'
                data =  re.sub(pattern, "url", data)
                
                # path_traversal ../../ 
                data =  re.sub(r' \.{1,} | \.\%\%\% ', " path_traversal ", data)
                
                data = re.sub(r'\.', ' ', data)
                
                data = re.sub(r'\\x00', ' ', data)
                
                data = re.sub(r" +", " ", data)
                data = data.lstrip()
                data = data.split(' ')
                
                """
                if header == "Path" or header == "Referer":
                    if "." in data[-1]:
                        name, ext = data[-1].split(".", maxsplit=1)
                        data[-1] = name
                        data.append(ext)
                """
                
                for i in range(len(data)):
                    if len(data[i]) > 200:
                        data[i] = "long_value"
                    
                payload[header] = data
                
            return payload

        def make_token(payload):
            tokens = []
            features = ["Method", "Host", "User-Agent", "Cookie", "Content-Length", "Content-Type", 
                        "Connection", "Accept", "Accept-Encoding", "Accept-Language", "Accept-Charset",
                        "X-Arachni-Scan-Seed", "Authorization", "Upgrade-Insecure-Requests", 
                        "X-Requested-With", "X-Forwarded-For", "Pragma", "Origin", "From", "Forwarded", "Range", 
                        "Referer", "Path", "Parameter", "Body"]
            
            def is_numeric_format(s):
                return bool(re.match(r'^\d+(\.\d+)?$', s))
            
            for h in features:
                if h in payload:
                    tokens.append(h)
                    #t = payload[h]
                    t = [item for item in payload[h] if not is_numeric_format(item)]
                    t = [x.lower() for x in t]
                    tokens += t

            tokens = [x for x in tokens if x != '']
            return tokens
                

        sentence_dict = preprocess(sentence)
        sentence_dict = preprocess2(sentence_dict)
        tokenized = make_token(sentence_dict)
        
        return tokenized
