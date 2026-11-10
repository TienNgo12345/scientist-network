// ===== Truy vấn mẫu & thuật toán đồ thị (Neo4j GDS) =====

// 1. Thống kê nút / cạnh theo loại
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY c DESC;
MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS c ORDER BY c DESC;

// 2. Phả hệ học thuật: chuỗi thầy-trò dài nhất bắt đầu từ một nhà khoa học
MATCH p = (a:Scientist {name:'Arnold Sommerfeld'})-[:DOCTORAL_ADVISOR_OF*1..4]->(s)
RETURN p LIMIT 50;

// 3. Ai bị phản biện / bác bỏ nhiều nhất
MATCH (a:Scientist)-[r:CRITICIZED|REFUTED]->(b:Scientist)
RETURN b.name, count(r) AS times ORDER BY times DESC LIMIT 10;

// 4. Bằng chứng (câu văn) của một quan hệ
MATCH (a)-[r:BUILT_UPON]->(b) RETURN a.name, b.name, r.evidence LIMIT 20;

// 5. Chiếu đồ thị nhà khoa học cho GDS
CALL gds.graph.project('sci', 'Scientist',
  {DOCTORAL_ADVISOR_OF:{orientation:'UNDIRECTED'}, ACADEMIC_ADVISOR_OF:{orientation:'UNDIRECTED'},
   INFLUENCED:{orientation:'UNDIRECTED'}, BUILT_UPON:{orientation:'UNDIRECTED'},
   CRITICIZED:{orientation:'UNDIRECTED'}, REFUTED:{orientation:'UNDIRECTED'},
   PRIORITY_DISPUTE:{orientation:'UNDIRECTED'}, COLLABORATED_WITH:{orientation:'UNDIRECTED'}});

// 6. PageRank, Betweenness, Louvain
CALL gds.pageRank.stream('sci') YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score ORDER BY score DESC LIMIT 10;
CALL gds.betweenness.stream('sci') YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score ORDER BY score DESC LIMIT 10;
CALL gds.louvain.stats('sci') YIELD communityCount, modularity;

// 7. Cộng đồng thông qua viện nghiên cứu: các nhà khoa học cùng làm việc tại một viện
MATCH (a:Scientist)-[:WORKED_AT]->(i:Institution)<-[:WORKED_AT]-(b:Scientist)
WHERE a.name < b.name RETURN i.name, count(*) AS pairs ORDER BY pairs DESC LIMIT 10;
