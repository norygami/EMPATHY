# EMPATHY

`EMPATHY` is an extension of [discollama](https://github.com/mxyng/discollama), a python script to prompt local LLMs mounted on [Ollama](https://github.com/jmorganca/ollama) through Discord bots. Primary additions include

## Primary Additions

`EMPATHY` significantly enhances the original `discollama` with the following key features:

* **Multi-Bot Support:** Configure and run multiple Discord bots simultaneously from a single instance using a `bot_config.json` file. Each bot can have its own Ollama model, name, and activity status.
* **Intelligent Thread Management:**
    * Automatically creates dedicated public threads for new conversations when the bot is mentioned in a text channel.
    * Manages thread ownership and prevents "trespassers" by allowing only the original thread creator to interact within their dedicated thread.
* **Enhanced Conversation Memory and Control:**
    * Maintains a more robust conversation history by storing full message roles and content in Redis.
    * Introduces a `/forget` command, allowing users to clear their conversation history with the bot at any time.
    * Includes a "system message" that helps guide the LLM's personality and behavior, dynamically updated in conversation history.
* **Improved Response Handling:**
    * Cleans and refines LLM output by removing unwanted tokens (e.g., ".assistant").
    * Intelligently splits long responses to comply with Discord's message length limits, indicating continuation with "...".
    * Edits previous bot messages instead of sending new ones where appropriate, providing a smoother and less cluttered chat experience.
* **Robust LLM Interaction:**
    * Utilizes Ollama's `chat` API for more natural, multi-turn conversational interactions.
    * Implements comprehensive timeout and error handling for Ollama API requests, ensuring bot resilience.
* **Performance Monitoring & Debugging:**
    * Integrates detailed logging for bot activities, including performance metrics like response time, response length, and history length.
    * Includes a "maintenance mode" with a user whitelist for controlled access during updates or debugging.

## Dependencies

-   Docker and Docker Compose

## Setup and Running EMPATHY

### 1. Configuration File (`bot_config.json`)

`EMPATHY` uses a `bot_config.json` file to manage configurations for one or more bots. Create this file in the root directory of your project. Here's an example structure:

```json
{
  "bots": [
    {
      "name": "MyEmpathyBot",
      "discord_bot_token": "DISCORD_TOKEN_ENV_VAR",
      "ollama": {
        "scheme": "http",
        "host": "127.0.0.1",
        "port": 11434
      },
      "redis": {
        "host": "127.0.0.1",
        "port": 6379
      },
      "model": "llama2",
      "activity": {
        "name": "with feelings",
        "state": "Empathizing with users"
      }
    },
    {
      "name": "AnotherBot",
      "discord_bot_token": "ANOTHER_DISCORD_TOKEN_ENV_VAR",
      "ollama": {
        "scheme": "http",
        "host": "127.0.0.1",
        "port": 11434
      },
      "redis": {
        "host": "127.0.0.1",
        "port": 6379
      },
      "model": "dolphin-mistral",
      "activity": {
        "name": "deep thoughts",
        "state": "Pondering existence"
      }
    }
  ]
}
