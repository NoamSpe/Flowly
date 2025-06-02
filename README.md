# About
Final School Software Engineering Project - NER/AI-based Task Management App

# Running Instructions
1. Download the project files.
2. Install the dependencies.
3. On the server's computer, adjust the ssl.conf file and regenerate the certificate and private key.
4. Modify the certificate "server.crt" on the client's computer to the newly generated certificate.
5. Run Server.py on the host computer.
6. Change the SERVER_HOST variable in "app_config.py" at the client's computer to the host computer's IP address.
7. Run FlowlyApp.py on the client computer.

# Firewall Settings (if needed)
Windows Defender Firewall -> Advanced settings -> Inbound Rules -> New Rule... -> Port -> TCP -> Specific local ports: 4320 -> Allow the connection.