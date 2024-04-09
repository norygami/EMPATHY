import io
import re
import os
import subprocess
import json
from enum import Enum, auto
import asyncio
import argparse
from datetime import datetime, timedelta
import ollama
import discord
import redis
from logger import LogData, log_performance
from time import perf_counter
import logging
import threading

# Configure the root logger
logging.basicConfig(level=logging.INFO)


# Redis init
command = ["docker", "start", "redis-stack-server"]
result = subprocess.run(command, capture_output=True, text=True)

# piggy back on the logger discord.py set up
logging = logging.getLogger('discord.discollama')

#Debug Context Packages
debug_context = False

class Response:
  def __init__(self, message, redis_client, bot_user_id, bot_name):
    self.message = message
    self.channel = message.channel
    self.bot_user_id = bot_user_id
    self.bot_name = bot_name
    self.redis = redis_client
    self.r = None
    self.sb = io.StringIO()
    self.last_react_time = datetime.min


  async def write(self, s):
      # Define the unwanted token
      print(f'PrepostProcess: {s}')
      s = self._remove_unwanted_fragments(s)
      unwanted_tokens = ["<|im_end|>", "<|im_start|>"]
      # Remove the unwanted token from 's' if it exists
      for token in unwanted_tokens:
        s = s.replace(token, "").rstrip()
      # Write the new content to the buffer
      print(f'PostPostProcess: {s}')
      self.sb.write(s)
      
      # Check if the current buffer exceeds the Discord message length limit
      if self.sb.seek(0, io.SEEK_END) > 2000:
          # Get the current buffer's content and reset it for the next message part
          value = self.sb.getvalue().strip()
          self.sb.seek(0, io.SEEK_SET)
          self.sb.truncate()

          # Split value into chunks that fit within Discord's limit
          chunk_size = 1900  # Use a slightly smaller size to account for 'end'
          chunks = [value[i:i+chunk_size] for i in range(0, len(value), chunk_size)]

          for i, chunk in enumerate(chunks):
            if i < len(chunks) - 1:
                chunk += '...'  # Indicate continuation
            await self._send_or_edit(chunk)
            self.r = None  # Reset self.r to ensure the next chunk is sent as a new message

      # Message length <2000 
      else:
          value = self.sb.getvalue().strip()
          if value:  # There's something to send
              await self._send_or_edit(value)
              # Reset the buffer after sending/editing
              self.sb.seek(0, io.SEEK_SET)
              self.sb.truncate()

              
  def _remove_unwanted_fragments(self, text):
    pattern = r"(user|assistant).*?(?=\n+system|$)"

    # Remove these patterns from the text
    # flags=re.DOTALL is used to make the '.' special character match any character at all, including a newline
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text.strip()

  async def _send_or_edit(self, value):
      if self.r:
          await self.r.edit(content=value)
      else:
          if self.channel.type == discord.ChannelType.text:
              thread_name = f'{self.bot_name}'
              self.channel = await self.channel.create_thread(name=thread_name, message=self.message, auto_archive_duration=60)
          self.r = await self.channel.send(value)

class reactStatus(Enum):
    INIT = auto()
    ADDED = auto()
    REMOVED = auto()
    CANCELLED = auto()
    NULL = auto()
    FAIL = auto()
    RATE = auto()

class Discollama:
  def __init__(self, ollama, discord, redis, model, use_reactions=False, activity=None, bot_name=None):
    self.ollama = ollama
    self.discord = discord
    self.redis = redis
    self.model = model
    self.use_reactions = use_reactions 
    self.activity_config = activity
    self.bot_name = bot_name
    self.last_react_time = datetime.min

    # register event handlers
    self.discord.event(self.on_ready)
    self.discord.event(self.on_message)

  async def on_ready(self):
      if self.activity_config:
          #activity_type = getattr(discord.ActivityType, self.activity_config['type'], discord.ActivityType.playing)
          activity = discord.Activity(name=self.activity_config['name'], type=discord.ActivityType.custom)
          if 'state' in self.activity_config:
              activity.state = self.activity_config['state']
              print(activity.state)
          await self.discord.change_presence(activity=activity)
      
      logging.info(f" {self.bot_name} is ready and online!")

  async def on_message(self, message):
    start_time = perf_counter()  # Start timer

    logging.info(f"Message from {message.author.id}: {message.content}")

    if message.author.bot:
        logging.info("Message from a bot. Ignoring.")
        return  # Ignore messages from any bot, including itself

    # Check if the message is a DM or the bot is mentioned directly
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_bot_mentioned = self.discord.user.mentioned_in(message)

    is_in_owned_thread = False
    if message.channel.type == discord.ChannelType.public_thread:
        is_in_owned_thread = message.channel.owner_id == self.discord.user.id

    # Log the context of the message
    context_msg = ('Bot mentioned' if is_bot_mentioned else
                   'DM' if is_dm else
                   'Owns Thread' if is_in_owned_thread else 
                   'Other')
    logging.info(f"Bot ID: {self.discord.user.id}. Context: {context_msg}")

    # Proceed if the message is in DM, the bot is mentioned, or it's in an owned thread
    if not (is_dm or is_bot_mentioned or is_in_owned_thread):
        logging.info("Ignoring message not in DM, not a mention, and not in an owned thread.")
        return
    
    if is_dm and message.content.strip().lower() == '/forget':
        await self.forget_conversation(message.author.id)
        await message.channel.send("\n**Here's to a new us! 🥂**")
        return

    # Handle the message
    content = message.content.replace(f'<@{self.discord.user.id}>', '').strip() if is_bot_mentioned else message.content
    if not content:
        content = 'Hi!'  # Default response if message content is empty after stripping mention

    channel = message.channel

    context = []
    if reference := message.reference:
      context = await self.load(message_id=reference.message_id)
      if not context:
        reference_message = await message.channel.fetch_message(reference.message_id)
        content = '\n'.join(
          [
            content,
            'Use this to answer the question if it is relevant, otherwise ignore it:',
            reference_message.content,
          ]
        )
    
    if not context:
      context = await self.load(channel_id=channel.id if not is_dm else None, user_id=message.author.id if is_dm else None)
    #print('Context Length: ' + (str(len(context))))

    r = Response(message, self.redis, self.discord.user.id, self.bot_name)
    if self.use_reactions:
      react_task = asyncio.create_task(self.react(message))
    
    async with message.channel.typing():
        # Generate and send response
        async for part in self.generate(content, context):
            # Check task status before cancellation
            if self.use_reactions and not react_task.done():
                react_task.cancel()
                try:
                    await react_task  # Allow any cleanup in the task to complete
                except asyncio.CancelledError:
                    pass  # Expected behavior upon task cancellation
            logging.info(f"About to send response: {part['response'][:50]}-")
            await r.write(part['response'])

        await r.write('')

        # Calculate response time
        end_time = perf_counter()
        response_time = end_time - start_time

        #print('Context Length: ' + (str(len(context))))

        # Logging
        log_data = LogData(
            user_channel_id=message.author.id if isinstance(message.channel, discord.DMChannel) else message.channel.id,
            context_length=len(context),
            response_time=response_time,
            prompt_length=len(content),
            response_length=len(part['response']),
            react_status=self.react_status.name if self.use_reactions else None,
        )
        log_performance(log_data)

        await self.save(r.channel.id, message.id, part['context'], message.author.id if is_dm else None)


  async def react(self, message, timeout=999):
      if not self.use_reactions:
        return
      self.react_status = reactStatus.INIT
      cooldown_period = timedelta(seconds=10)  # 10 seconds for example
      if datetime.now() - self.last_react_time < cooldown_period:
          return  # Skip if we're within the cooldown period

      try:
          await message.add_reaction('🤔')
          self.last_react_time = datetime.now()
          self.react_status = reactStatus.ADDED
          
          await asyncio.sleep(timeout)
      except asyncio.CancelledError:
          # This block is executed if the task is cancelled
          self.react_status = reactStatus.CANCELLED
      except Exception as e:
          logging.warning(f"Encountered exception in thinking: {e}")
      finally:
          try:
              await message.remove_reaction('🤔', self.discord.user)
          except discord.NotFound:
              # This may happen if the message was deleted before the react could be removed
              self.react_status = reactStatus.NULL

          except discord.RateLimited:
            # This may happen if the message was deleted before the react could be removed
             self.react_status = reactStatus.RATE
          except Exception as e:
              self.react_status = reactStatus.FAIL
          else:
            # Only set to REMOVED if no exceptions were raised during removal
            self.react_status = reactStatus.REMOVED

  async def generate(self, content, context_ids):
      # Modify the formatted prompt to include the bot_name as the character's name in the system layer
      formatted_prompt = f"system\n{self.bot_name}\nuser\n{content}\nassistant\n"
      print(f'Formatted prompt: {formatted_prompt}')

      async def request_with_timeout():
          try:
              # Assuming `ollama.generate` returns an awaitable object (response)
              response = await self.ollama.generate(model=self.model, prompt=formatted_prompt, context=context_ids, keep_alive=-1, stream=False)
              return response
          except asyncio.TimeoutError:
              logging.warning("Timeout")

              # Perform any necessary cleanup here
              return {"response": "Request timed out.", "context": []}
          except Exception as e:
              logging.exception("Exception in generate:", exc_info=e)
              return {"response": "Error generating response.", "context": []}

      # Adjust the timeout as needed
      response = await asyncio.wait_for(request_with_timeout(), timeout=60.0)
      yield response

  async def save(self, channel_id, message_id, ctx: list[int], user_id=None):
      key_prefix = f'discollama:{self.discord.user.id}:dm:' if user_id else f'discollama:{self.discord.user.id}:channel:'
      key = f"{key_prefix}{user_id or channel_id}"
      self.redis.set(key, json.dumps(message_id), ex=60 * 60 * 24 * 7)
      self.redis.set(f'discollama:message:{message_id}', json.dumps(ctx), ex=60 * 60 * 24 * 7)
      #print(f"Saved updated context IDs for key {key}: {json.dumps(ctx)}")

      # Track message IDs in a set for DM conversations
      if user_id:
          messages_set_key = f'discollama:{self.discord.user.id}:user_messages:{user_id}'
          self.redis.sadd(messages_set_key, message_id)
          # Optionally set an expiration for the set itself, though this may depend on your application's needs
          self.redis.expire(messages_set_key, 60 * 60 * 24 * 7)

  async def load(self, channel_id=None, user_id=None) -> list[int]:
      message_id = None
      key_prefix = f'discollama:{self.discord.user.id}:dm:' if user_id else f'discollama:{self.discord.user.id}:channel:'
      key = f"{key_prefix}{user_id or channel_id}"
      print(f"Loading context for key: {key}")
      message_id = self.redis.get(key)
      ctx = self.redis.get(f'discollama:message:{message_id}')
      return json.loads(ctx) if ctx else []
  
  async def forget_conversation(self, user_id):
      # Retrieve the set of all message IDs for the user
      message_ids_set_key = f'discollama:{self.discord.user.id}:user_messages:{user_id}'
      message_ids = self.redis.smembers(message_ids_set_key)

      # Delete each message context key
      for msg_id in message_ids:
          self.redis.delete(f'discollama:message:{msg_id}')

      # Delete the set itself
      self.redis.delete(message_ids_set_key)

      # Continue with deleting the latest message key as before
      latest_message_key = f'discollama:{self.discord.user.id}:dm:{user_id}'
      self.redis.delete(latest_message_key)
  
  def run(self, token):
    try:
      self.discord.run(token)
    except Exception:
      self.redis.close()

def load_config():
    with open('bot_config.json', 'r', encoding='utf-8') as config_file:
        return json.load(config_file)
    
async def start_bot(bot_config):
    intents = discord.Intents.default()
    intents.message_content = True
    
    discord_client = discord.Client(intents=intents)
    
    bot = Discollama(
        ollama.AsyncClient(
            host=f"{bot_config['ollama']['scheme']}://{bot_config['ollama']['host']}:{bot_config['ollama']['port']}"
        ),
        discord_client,
        redis.Redis(
            host=bot_config['redis']['host'], 
            port=bot_config['redis']['port'], 
            db=0, 
            decode_responses=True
        ),
        model='TRACHI',
        use_reactions=bot_config['use_reactions'],
        activity=bot_config.get('activity'),
        bot_name=bot_config['name']
    )
    
    await discord_client.login(os.environ[bot_config['discord_bot_token']])
    await discord_client.connect()

async def main():
  config = load_config()
  bot_tasks = [start_bot(bot_config) for bot_config in config['bots']]
  await asyncio.gather(*bot_tasks)

if __name__ == '__main__':
    asyncio.run(main())
