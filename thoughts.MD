tools:
- is there a datawrangler from json or smthg similar? good for checking distribution of text length, mentions etc.

data:
- relation_schema.json: allowed subject/object relation schema derived *from training data* - correct, checks out with kaggle schema
- mention_model/config.json:
    - explain all settings
       - these are the model settings, not our hp! those are in experiment_config.json
- mention_model/tokenizer.json:
    - what's with all the extra tokens?
    - what does this actually do? Where is it used? How is the vocabulary built?
    - why do words start with   "Ġ?
        - this is all preset in the model, that is a marker for start of word
        - our only modification is some added tokens (to the relation model, not the mention model)
- dev_predicted_mentions.json:
    - !!!!! extremely low number of predictions (8), when does this get built in the pipeline?
        - after training the mention model on train and running it on dev (as validation) - this is the dev run
    - also most of them are clearly wrong - just 1 TP (pest), all the rest are FP
        - sign that the mention model fails miserably
- dev_predicted_entities.json:
    - what are the entity_id fields? why are they repeating? 
        - looks like they are per document
- relation_model/config.json:
    - num_attention_heads, num_hidden_layers: are these the model params or hyperparams? why these values if hp?
        - they are model settings, not our hp!
- same questions as other model with the tokenizer.json
    - same answers, mostly stock params
- experiment_config.json
    - walk through parameter choice reasoning