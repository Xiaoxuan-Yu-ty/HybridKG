# Summarized documentation for managing Neo4j container and data lifecycle.

### 1. Project Directory Setup

Keep your files organized to ensure Docker can map volumes correctly. Your host directory should look like this:

```
/project-root/
├── docker-compose.yml
├── data/           # Stores Neo4j database files
├── import/         # Place your .csv files here for imports
└── plugins/        # Place APOC .jar files here (if not downloading automatically)
```

### 2. The`docker-compose.yml` Configuration

This configuration ensures APOC is enabled, permissions are handled, and volumes are mounted.

```YAML
services:
  neo4j:
    image: neo4j:latest # my laptop only support this image(quay.io/lib/neo4j:community-bullseye)
    container_name: <self-define>
    ports:
      - "7474:7474" # Browser (self-define)
      - "7687:7687" # Bolt (self-define))
    environment:
      NEO4J_AUTH: neo4j/<your_password>
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_apoc_export_file_enabled: "true"
      NEO4J_apoc_import_file_enabled: "true"
      NEO4J_dbms_security_procedures_allowlist: "apoc.*"
      NEO4J_dbms_security_procedures_unrestricted: "apoc.*"
    volumes:
      - ./data:/data
      - ./import:/var/lib/neo4j/import
      - ./plugins:/var/lib/neo4j/plugins
```

### 3. Importing `.csv`with `neo4j-admin`

The`neo4j-admin import` tool is for ****bulk loading** massive datasets. It must be run while the database is** **stopped** .

1. **Stop the container:** `docker-compose down`
2. **Run the import command:**
   **Bash**

   ```Shell
   docker run --rm \
     -v $PWD/data:/data \
     -v $PWD/import:/var/lib/neo4j/import \
     neo4j:latest \
     neo4j-admin database import full --nodes=import/nodes.csv --relationships=import/edges.csv
   ```

   *Note: Replace** **`full` with your specific database name (default is usually** **`neo4j`).*

### 4. Exporting to a`.cypher` File

Once your database is running and APOC is installed, use the APOC procedure in the Neo4j Browser or`cypher-shell`.

**To export the entire database:**

**Cypher**

```
CALL apoc.export.cypher.all("full_graph.cypher", {
    format: "cypher-shell",
    useOptimizations: {type: "UNWIND_BATCH", unwindBatchSize: 2000}
})
```

* The file **`full_graph.cypher` will appear in your local**`./import/` folder.

### Summary Checklist for Future Use

* **Permissions:** If you encounter`Permission Denied` errors, ensure your local directory is owned by your user (`sudo chown -R $USER:$USER ./import`).
* **Plugins:** Always verify plugin loading with:`SHOW PROCEDURES YIELD name WHERE name STARTS WITH "apoc"`.
* **Clean Slate:** If you need to restart from scratch, run`docker-compose down -v` to wipe the volumes and container configuration.
