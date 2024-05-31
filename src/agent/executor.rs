use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use serde_json::json;
use tokio::sync::Mutex;

use crate::{
    chain::{chain_trait::Chain, ChainError},
    language_models::GenerateResult,
    memory::SimpleMemory,
    prompt::PromptArgs,
    schemas::{
        agent::{AgentAction, AgentEvent},
        memory::BaseMemory,
    },
    tools::Tool,
};

use super::{agent::Agent, AgentError};

pub struct AgentExecutor<A>
where
    A: Agent,
{
    agent: A,
    max_iterations: Option<i32>,
    break_if_error: bool,
    pub memory: Option<Arc<Mutex<dyn BaseMemory>>>,
}

impl<A> AgentExecutor<A>
where
    A: Agent,
{
    pub fn from_agent(agent: A) -> Self {
        Self {
            agent,
            max_iterations: Some(10),
            break_if_error: false,
            memory: None,
        }
    }

    pub fn with_max_iterations(mut self, max_iterations: i32) -> Self {
        self.max_iterations = Some(max_iterations);
        self
    }

    pub fn with_memory(mut self, memory: Arc<Mutex<dyn BaseMemory>>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn with_break_if_error(mut self, break_if_error: bool) -> Self {
        self.break_if_error = break_if_error;
        self
    }

    fn get_name_to_tools(&self) -> HashMap<String, Arc<dyn Tool>> {
        let mut name_to_tool = HashMap::new();
        for tool in self.agent.get_tools().iter() {
            log::info!("Loading Tool:{}", tool.name());
            name_to_tool.insert(tool.name().trim().replace(" ", "_"), tool.clone());
        }
        name_to_tool
    }
}

#[async_trait]
impl<A> Chain for AgentExecutor<A>
where
    A: Agent + Send + Sync,
{
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError> {
        let mut input_variables = input_variables.clone();
        let name_to_tools = self.get_name_to_tools();
        let mut steps: Vec<(AgentAction, String)> = Vec::new();
        log::debug!("steps: {:?}", steps);
        if let Some(memory) = &self.memory {
            let memory = memory.lock().await;
            input_variables.insert("chat_history".to_string(), json!(memory.messages()));
        } else {
            input_variables.insert(
                "chat_history".to_string(),
                json!(SimpleMemory::new().messages()),
            );
        }

        loop {
            let agent_event = self
                .agent
                .plan(&steps, input_variables.clone())
                .await
                .map_err(|e| {
                    ChainError::AgentError(format!("Error in agent planning: {}", e.to_string()))
                })?;
            match agent_event {
                AgentEvent::Action(actions) => {
                    for action in actions {
                        log::debug!("Action: {:?}", action.tool_input);
                        let tool = name_to_tools
                            .get(&action.tool)
                            .ok_or_else(|| {
                                AgentError::ToolError(format!("Tool {} not found", action.tool))
                            })
                            .map_err(|e| ChainError::AgentError(e.to_string()))?;

                        let observation_result = tool.call(&action.tool_input).await;

                        let observation = match observation_result {
                            Ok(result) => result,
                            Err(err) => {
                                log::info!(
                                    "The tool return the following error: {}",
                                    err.to_string()
                                );
                                if self.break_if_error {
                                    return Err(ChainError::AgentError(
                                        AgentError::ToolError(err.to_string()).to_string(),
                                    ));
                                } else {
                                    format!(
                                        "The tool return the following error: {}",
                                        err.to_string()
                                    )
                                }
                            }
                        };

                        steps.push((action, observation));
                    }
                }
                AgentEvent::Finish(finish) => {
                    if let Some(memory) = &self.memory {
                        let mut memory = memory.lock().await;
                        memory.add_user_message(&input_variables["input"]);
                        memory.add_ai_message(&finish.output);
                    }
                    return Ok(GenerateResult {
                        generation: finish.output,
                        ..Default::default()
                    });
                }
            }

            if let Some(max_iterations) = self.max_iterations {
                if steps.len() >= max_iterations as usize {
                    return Ok(GenerateResult {
                        generation: "Max iterations reached".to_string(),
                        ..Default::default()
                    });
                }
            }
        }
    }

    async fn invoke(&self, input_variables: PromptArgs) -> Result<String, ChainError> {
        let result = self.call(input_variables).await?;
        Ok(result.generation)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        agent::ConversationalAgentBuilder, chain::options::ChainCallOptions,
        language_models::llm::LLM, llm::client::Ollama, prompt_args, schemas::Message,
        tools::CommandExecutor,
    };

    use super::*;
    use tokio::io::AsyncWriteExt;
    use tokio_stream::StreamExt;

    #[tokio::test]
    async fn test_agent() {
        let llm = Ollama::default().with_model("llama3");
        let memory = SimpleMemory::new();
        let command_executor = CommandExecutor::default();
        let agent = ConversationalAgentBuilder::new()
            .tools(&[Arc::new(command_executor)])
            .options(ChainCallOptions::new().with_max_tokens(1000))
            .build(llm)
            .unwrap();

        let executor = AgentExecutor::from_agent(agent).with_memory(memory.into());

        let input_variables = prompt_args! {
            "input" => "What is the name of the current dir",
        };

        match executor.invoke(input_variables).await {
            Ok(result) => {
                println!("Result: {:?}", result);
            }
            Err(e) => panic!("Error invoking LLMChain: {:?}", e),
        }
    }
    #[tokio::test]
    #[ignore]
    async fn test_generate() {
        let ollama = Ollama::default().with_model("llama3");
        let response = ollama.invoke("Hey Macarena, ay").await.unwrap();
        println!("{}", response);
    }

    #[tokio::test]
    async fn test_stream() {
        let ollama = Ollama::default().with_model("llama3");

        let message = Message::new_human_message("Why does water boil at 100 degrees?");
        let mut stream = ollama.stream(&vec![message]).await.unwrap();
        let mut stdout = tokio::io::stdout();
        while let Some(res) = stream.next().await {
            let data = res.unwrap();
            stdout.write(data.content.as_bytes()).await.unwrap();
        }
        stdout.write(b"\n").await.unwrap();
        stdout.flush().await.unwrap();
    }
}
