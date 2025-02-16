llm_config_dict = {'generate-summary': {"task" : "text-generation",
                      "temperature" : 0.0000000001,
                      "max_new_tokens" : 200,
                      "model_max_length" : 8500,
                      "max_summ_words" : 150},
                      
                    'evaluate-summary':{"task" : "text-generation",
                                          "temperature" : 0.0000000001,
                                          "max_new_tokens" : 400
                                          },

                    'improve-generate-summary': {"task" : "text-generation",
                      "temperature" : 0.0000000001,
                      "max_new_tokens" : 200,
                      "model_max_length" : 8500,
                      "max_summ_words" : 150},

                    'improve-evaluate-summary':{"task" : "text-generation",
                                        "temperature" : 0.0000000001,
                                        "max_new_tokens" : 400
                                        },
 
}