# About
Final School Software Engineering Project - NER/AI-based Task Management App

# Pay Attention
* When changing the SERVER_HOST variable or any variable in app_config remember to save the file before running the client again

# Running Instructions
-- on both computers
1. Download the project files.
2. Install the dependencies.
3. Set up openssl ('SSL Establishment' bellow).
--
3. On the server's computer, adjust the ssl.conf file and regenerate the certificate and private key with openssl.
4. Modify the certificate "server.crt" on the client's computer to the newly generated certificate.
5. Run Server.py on the host computer.
6. Change the SERVER_HOST variable in "app_config.py" at the client's computer to the host computer's IP address, and save the file.
7. Run FlowlyApp.py on the client computer.

# Firewall Settings (if needed)
Windows Defender Firewall -> Advanced settings -> Inbound Rules -> New Rule... -> Port -> TCP -> Specific local ports: 4320 -> Allow the connection.

# SSL Establishment (certificate and private key generation with OpenSSL)
* Download installer from: slproweb.com/products/Win32OpenSSL.html
* Install OpenSSL to your computer
* Copy the path to the "bin" folder in the installed OpenSSL directory
## set the environment path
* Go To:
system -> environment variables -> system variables -> Path (Edit)
* Copy the path to the "bin" folder in the installed OpenSSL directory
* Add new environment variable
* Paste the path