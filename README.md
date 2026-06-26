# HybridKG

## Project Structure

```
--HYBRIDKG
 |-rmp_models 
 | |-__init__.py
 | |-__pycache__
 | |-train_rmp.py
 | |-hetero_rmp_models.py
 |-gnn_clep
 | |-shgt_hits@k.py
 | |-peter.sh
 | |-__init__.py
 | |-__pycache__
 | |-model.py
 | |-ML_Classifier
 | |-peter_logs
 | |-train.py
 | |-train_gnnclep.py
 | |-test.ipynb
 |-GateEmbeddingTask
 | |-train_utils.py
 | |-explore.ipynb
 | |-TwoStageMLT
 | |-__init__.py
 | |-encoders.py
 | |-__pycache__
 | |-torch.md
 | |-HRGNN
 | |-test_encoders.ipynb
 | |-HRGATConv.py
 | |-hetero_gate_model
 | |-HRGCNConv.py
 |-CLEP_repeat
 | |-classification
 | |-constants.py
 | |-embedding
 | |-retrain_clep_source.py
 | |-__init__.py
 | |-__pycache__
 | |-visualize.py
 | |-cls_pipeline.py
 | |-modify_cls.ipynb
 | |-cli.py
 | |-visualize.ipynb
 | |-peter_logs
 | |-pipeline.py
 | |-retrain_pipeline.py
 | |-leo_pipeline.sh
 | |-clep_resources
 | |-sample_scoring
 | |-repeat_resource.ipynb
 | |-test.ipynb
 | |-__main__.py
 | |-peter_cls_pipeline.sh
 |-datasets
 | |-base_kgs
 | |-FireGNN_KGs
 | |-TrainSample_KGs
 | |-ADNI_KGs
 |-EdgeAssignmentTask
 | |-__init__.py
 | |-hetero_base_models
 | |-lp_edge_assignment
 |-data_processing
 | |-pyg_graph_utils.py
 | |-build_graphs.py
 | |-__init__.py
 | |-pyg_graph_generator.py
 | |-__pycache__
 | |-patient_network_prep.py
 | |-network_generator.py
 | |-build_patient_graphs.py
 | |-sample_scoring.py
 | |-test_convert_hetero.ipynb
 | |-pyg_graph_generate.sh
 | |-build_morpho_graphs.py
 | |-geo_processing.py
 | |-graph_generate_pipeline.py
 |-__init__.py
 |-utils
 | |-add_difference_nodes.py
 | |-graph_utils.py
 | |-__init__.py
 | |-__pycache__
 | |-visualize.py
 | |-helper.py
 |-ML_Classifier
 | |-__init__.py
 | |-utils.py
 | |-hpo.py
 |-PyKeen
 | |-hpo_train_cls.ipynb
 | |-run_linkPrediction.sh
 | |-run_train_cls.sh
 | |-__init__.py
 | |-__pycache__
 | |-results
 | |-thoughts_notes.md
 | |-configs
 | |-link_prediction.py
 | |-train_cls.py
 | |-run_hpo.sh
 | |-edge_assignment.ipynb
 | |-hpo.py
 |-FuzzyGNN
 | |-__init__.py
 | |-gnn_models
 | |-fuzzy_gnn_models
```
