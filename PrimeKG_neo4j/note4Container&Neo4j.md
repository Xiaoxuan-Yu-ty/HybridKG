# Notes of using docker and neo4j

### Multiple container

- docker-compose.yml file: can write multiple graph container

  ```
  services:

    neo4j-kg1:
      image: quay.io/lib/neo4j:community-bullseye
      container_name: neo4j_commute
      ports:
        - "7474:7474"   # Browser
        - "7687:7687"   # Bolt
      environment:
        - NEO4J_AUTH=neo4j/test1234
      volumes:
        - ./data/neo4j_kg1:/data
        - ./imports/import_commute:/var/lib/neo4j/import

    neo4j-kg2:
      image: quay.io/lib/neo4j:community-bullseye
      container_name: neo4j_healthy
      ports:
        - "7475:7474"   # Browser (different host port!)
        - "7688:7687"   # Bolt (different host port!)
      environment:
        - NEO4J_AUTH=neo4j/test1235
      volumes:
        - ./data/neo4j_kg2:/data
        - ./imports/import_healthy:/var/lib/neo4j/import
  ```

  1. Port Mapping (Host:Container)
     + Containers live in their own network bubble.
     + For the first container: 7474:7474 means you go to localhost:7474.
     + For the second container: 7475:7474 means you go to localhost:7475.
     + Important: Inside the container, Neo4j still thinks it's on 7474, but Docker acts as a translator.
  2. Neo4j localhost Mapping:
     + When open the brower, use localhost:747x to land the neo4j login page;
     + choose neo4j://localhost:768x accordingly to log in the target Neo4j database.
  3. Unique Volume Names
     + If both containers used the volume neo4j:/data, they would try to write to the same database files, causing a crash or data corruption. I renamed them to neo4j_data_organic and neo4j_data_kg.
  4. Container Names
     + The container_name property ensures that when you run docker ps, you can easily tell which graph is which.
  5. Execuation:
     + in terminal under docker-compose.yml directory, run `docker-compose up -d` can start all the defined containers
     + open localhost:747x (or other port) can open neo4j browser
     + choose neo4j://localhost: 768x bolt to connect to related container database.

  - directory setup
    neo4j-project/
    ├── docker-compose.yml
    ├── data/
    │            ├── neo4j_kg1/ (related data files will be saved by neo4j afterwards)
    │            ├── neo4j_kg2/
    │            └── neo4j_kg3/
    └── imports/
    ├── kg1/ (put .cypher KG files here for later importing)
    ├── kg2/
    └── kg3/

### Upload KG to neo4j-database

#### Cypher file

+ In terminal run: `docker exec -i container_name cypher-shell -u neo4j -p test123456 < ./imports/kg_import_directory/kg_name.cypher`

#### pkl file

1. Write KG triples to neo4j database:
   - connect to localhost by a driver
   - load KG
   - write the triples to neo4j-database
2. Convert .pkl file to .csv file and then import

+ Here the .csv files of nodes and edges should follow certain format:
  + | node_id      | :ID      | :LABEL  |
    | :----------- | :------- | :------ |
    | HGNC:BDNF    | BDNF     | Gene    |
    | MESH:D003920 | Diabetes | Disease |
  + | :START_ID | :END_ID      | :TYPE           | confidence:float | evidence                            |
    | :-------- | :----------- | :-------------- | :--------------- | :---------------------------------- |
    | HGNC:BDNF | HGNC:TNF     | INHIBITS        | 0.95             | "Direct interaction observed in..." |
    | HGNC:TNF  | MESH:D003920 | ASSOCIATED_WITH | 0.88             | "GWAS study identifies linkage..."  |

3. Convert .pkl file to .cypher file and then import

#### CSV file

+ docker-compose.yml:

  + ```
    services:
      neo4j-ppikg:
        image: quay.io/lib/neo4j:community-bullseye
        container_name: neo4j_ppikg
        ports:
          - "7479:7474"
          - "7692:7687"
        environment:
          - NEO4J_AUTH=neo4j/test1234
          # Updated for Neo4j 5.x
          - NEO4J_server_memory_heap_initial__size=1G
          - NEO4J_server_memory_heap_max__size=2G
          - NEO4J_server_memory_pagecache_size=2G
          # Updated import setting
          - NEO4J_server_directories_import=import
          - NEO4J_server_config_strict__validation_enabled=false
        volumes:
          - ./data:/data
          - ./import:/import
          - ./plugins:/plugins
          - ./logs:/logs
        restart: alway
    ```


+ Set up neo4j on using docker/podman: `docker-compose up -d`
+ Stop neo4j container to use neo4j-admin tool: `docker stop neo4j`
+ import csv files (updated version):

  + `docker compose run --rm neo4j-primekg neo4j-admin database import full --nodes=/import/nodes.csv --relationships=/import/edges.csv --overwrite-destination --verbose`
+ Restart the container: `docker-compose up -d`
+ Explore the graph using Cypher
