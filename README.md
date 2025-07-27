# EMPATHY

`EMPATHY` is a prototypical extension of [discollama](https://github.com/mxyng/discollama), a python script to prompt [Ollama](https://github.com/jmorganca/ollama) models via Discord bots.

## Primary Additions
`EMPATHY` adds following:

* **Multi-Bot Support:** Configure and run multiple Discord bots simultaneously from a single instance using a `bot_config.json` file. Each bot can have its own Ollama model, name, and activity status.
* **Intelligent Thread Management:**
    * Automatically creates dedicated public threads for new conversations when the bot is mentioned in a text channel.
    * Manages thread ownership and prevents "trespassers" by allowing only the original thread creator to interact within their dedicated thread.
* **Enhanced Conversation Memory and Control:**
    * Maintains a comprehensive conversation history by storing full message roles and content in Redis.
    * Introduces a `/forget` command, allowing users to clear their conversation history with the bot at any time.
    * Includes a "system message" tied to specific bots.
* **Improved Response Handling:**
    * Additional post-processing (e.g., removing ".assistant").
    * Long responses are chunked to comply with Discord's message length limits.
* **Robust LLM Interaction:**
    * Utilizes Ollama's `chat` API for more natural, multi-turn conversational interactions.
    * Implements comprehensive timeout and error handling for Ollama API requests.
* **Performance Monitoring & Debugging:**
    * Integrates detailed logging for bot activities, including performance metrics like response time, response length, and history length.
    * Includes a "maintenance mode" with a user whitelist for controlled access during updates or debugging.

## Dependencies
-   Docker & Compose

## Setup and Run

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

````

  * `name`: The display name of your bot.
  * `discord_bot_token`: The name of the environment variable that holds your Discord bot token (e.g., `DISCORD_TOKEN`).
  * `ollama`: Configuration for the Ollama server connection.
  * `redis`: Configuration for the Redis server connection (used for conversation history).
  * `model`: The Ollama model to use for this specific bot instance.
  * `activity`: (Optional) Custom activity status for your bot.

### 2\. Discord Bot Setup

You must set up a [Discord Bot](https://discord.com/developers/applications) for each bot defined in your `bot_config.json` and obtain their respective tokens. These tokens should be set as environment variables as specified in your `bot_config.json` (e.g., `DISCORD_TOKEN=xxxxx`, `ANOTHER_DISCORD_TOKEN_ENV_VAR=yyyyy`).

### 3\. Ollama Server Setup

`EMPATHY` requires an [Ollama](https://github.com/ollama/ollama) server. Follow the steps in the [ollama/ollama](https://github.com/ollama/ollama) repository to set up Ollama.

By default, the `bot_config.json` uses `127.0.0.1:11434` for Ollama, which can be overridden in the configuration file if your Ollama instance is on a different host or port.

> **Note:** Deploying this on Linux might require updating network configurations and your Ollama host settings within `bot_config.json`.

### 4\. Running EMPATHY

To run `EMPATHY` using Docker Compose, ensure your environment variables (like `DISCORD_TOKEN`) are set, then execute:

```bash
DISCORD_TOKEN=your_token_here ANOTHER_DISCORD_TOKEN_ENV_VAR=another_token_here docker compose up
```

Replace `your_token_here` and `another_token_here` with your actual Discord bot tokens.

## Customize EMPATHY

### Ollama Models

The `model` parameter in `bot_config.json` specifies which Ollama model each bot instance will use. For example, `model: "llama2"`.

### Custom Personalities

To add a custom personality, you can change the `SYSTEM` instruction in an Ollama `Modelfile` and run `ollama create`:

```bash
ollama create mymodel -f Modelfile
```

Then, update the `model` field in your `bot_config.json` for the relevant bot to `mymodel`.

See [ollama/ollama](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) for more details on `Modelfile` customization.

## Activating the Bot

Discord users can interact with the bot in a few ways:

  * **Starting a New Conversation:** Mention the bot in a public text channel (e.g., `@MyEmpathyBot How are you?`). This will automatically create a new thread for the conversation.
  * **Continuing an Ongoing Conversation:** Reply directly to a previous bot message within a thread.
  * **Direct Message (DM):** Send a direct message to the bot.
  * **Forgetting Conversation History:** Type `/forget` in a conversation with the bot (either in a thread or DM) to clear the bot's memory of your previous interactions.
