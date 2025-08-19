import psycopg2
import numpy as np
from cuvs.neighbors import ivf_pq
from psycopg2.extras import execute_batch
from typing import List, Tuple
from ast import literal_eval
import cupy 

class VectorSearchEngine:
    def __init__(self, db_config: dict):
        """初始化向量搜索引擎
        
        Args:
            db_config: PostgreSQL 数据库配置字典
                      包含 dbname, user, password, host, port 等
        """
        self.db_config = db_config
        self.index = None
        self.vector_dim = None
        
    def load_data(self, table_name: str, id_col: str, vector_col: str) -> Tuple[List[int], np.ndarray]:
        """从 PostgreSQL 加载向量数据
        
        Args:
            table_name: 包含向量数据的表名
            id_col: ID列名
            vector_col: 向量列名
            
        Returns:
            元组(IDs, 向量数组)
        """
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor() as cur:
                # 获取向量维度
                cur.execute(f"SELECT {vector_col} FROM {table_name} LIMIT 1")
                self.vector_dim = len(literal_eval(cur.fetchone()[0]))
                
                # 获取所有数据
                cur.execute(f"SELECT {id_col}, {vector_col} FROM {table_name}")
                data = cur.fetchall()
                
                ids = [row[0] for row in data]
                
                liter = [literal_eval(row[1]) for row in data]
                
                vectors = np.array(liter, dtype=np.float32)
                
                return ids, vectors
        finally:
            conn.close()
    
    def build_index(self, vectors: np.ndarray, algorithm: str = "ivf_flat", metric: str = "euclidean", **kwargs):
        """构建向量索引
        
        Args:
            vectors: 向量数据 (numpy数组)
            algorithm: cuVS算法 (ivf_flat, brute, etc.)
            metric: 距离度量 (euclidean, cosine, etc.)
            kwargs: 算法特定参数
        """
        index_params=ivf_pq.IndexParams(n_lists=2,metric=metric)
        self.index = ivf_pq.build(index_params, vectors)
        #self.index.fit(vectors)
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[List[float], List[int]]:
        """执行向量搜索
        
        Args:
            query_vector: 查询向量
            k: 返回的最近邻数量
            
        Returns:
            元组(距离列表, ID列表)
        """
        if self.index is None:
            raise ValueError("索引未初始化，请先调用build_index()")
            
        if len(query_vector) != self.vector_dim:
            raise ValueError(f"查询向量维度({len(query_vector)})与索引维度({self.vector_dim})不匹配")
        
        search_params = ivf_pq.SearchParams(n_probes=1) 
        query_vector_gpu=cupy.asarray(np.array([query_vector],dtype=np.float32))
        print(query_vector_gpu)
        distances, indices = ivf_pq.search(search_params, self.index, query_vector_gpu, k)
        distances_cpu=cupy.asnumpy(distances)
        indices_cpu=cupy.asnumpy(indices)
        return distances_cpu[0], indices_cpu[0]
    
#main
if __name__ == "__main__":
    # 配置数据库连接
    import argparse

    parser = argparse.ArgumentParser(description="向量搜索引擎示例")
    parser.add_argument("--database", type=str, default="vector_db", help="数据库名")
    parser.add_argument("--user", type=str, default="vector_user", help="数据库用户名")
    parser.add_argument("--password", type=str, default="secure_password", help="数据库密码")
    parser.add_argument("--host", type=str, default="localhost", help="数据库主机")
    parser.add_argument("--port", type=int, default=5432, help="数据库端口")
    args = parser.parse_args()

    db_config = {
        "database": args.database,
        "user": args.user,
        "password": args.password,
        "host": args.host,
        "port": args.port
    }
    '''
    db_config = {
        "database": "vector_db",
        "user": "vector_user",
        "password": "secure_password",
        "host": "localhost",
        "port": 5432
    }
    '''
    # 初始化搜索引擎
    search_engine = VectorSearchEngine(db_config)
    
    # 从数据库加载数据
    ids, vectors = search_engine.load_data("items", "id", "embedding")
    
    # 构建索引 (可以调整IVF参数)
    search_engine.build_index(vectors, algorithm="ivf_flat", nlist=100)
    
    # 示例查询
    queries = cupy.random.random_sample((2, 3),
                                  dtype=cupy.float32)
    #print(queries)
    #print(search_engine.vector_dim)
    query_vec = np.random.rand(search_engine.vector_dim).astype(np.float32)
    #print(query_vec)
    distances, result_ids = search_engine.search(query_vec, k=1)
    
    print(f"最近邻IDs: {result_ids}")
    print(f"距离: {distances}")
    print(1)