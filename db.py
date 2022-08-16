from urllib.parse import quote

ssl_args = {'sslcert': '/persistent/postgres_cert/postgresql.crt',
            'sslkey': '/persistent/postgres_cert/postgresql.key'}
host    = '172.29.184.37'
dbname  = 'postgres'
user    = 'telenor_user'
port="5432"
password='Telenor@123'
server_str = 'postgresql+psycopg2://{0}:{1}@{2}:{3}/{4}'.format(user, quote(password),
                                                                host, port, dbname)

print(server_str)