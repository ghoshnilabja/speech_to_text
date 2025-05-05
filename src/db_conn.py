import psycopg2

port                  = '5432'

local_server_user     = 'postgres'
local_server_password = 'password'
local_server_host     = 'localhost'
local_server_database = 'nlp'

tick_server_user     = 'read_user'
tick_server_password = 'Fsread@123'
tick_server_host     = '192.168.29.85'
tick_server_database = 'postgres'

live_server_user     = 'read_user'
live_server_password = 'Vista@tick'
live_server_host     = '192.168.29.74'
live_server_database = 'postgres'




def getTickConnection(user=tick_server_user,password=tick_server_password,host=tick_server_host):    
    conn = psycopg2.connect(dbname = tick_server_database,
                            user = user,
                            password = password,
                            host = host,
                            port = port)
    return conn



def getLocalConnection(user=local_server_user, password=local_server_password, host=local_server_host,):   
    conn  =  psycopg2.connect(dbname = local_server_database,
                              user = user,
                              password = password,
                              host = host,
                              port = port)
    return conn

def getLiveConnection(user=live_server_user, password=live_server_password, host=live_server_host):
    conn  =  psycopg2.connect(dbname = live_server_database,
                              user = user,
                              password = password,
                              host = host,
                              port = port)
    return conn


def closeConnection(conn):
    conn.close()