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

# piggy back on the logger discord.py set up
logging = logging.getLogger('discord.discollama')

#Debug Context Packages
debug_context = False

class Response:
  def __init__(self, message, channel, redis_client, bot_user_id, bot_name):
    self.message = message
    self.channel = channel
    self.bot_user_id = bot_user_id
    self.bot_name = bot_name
    self.redis = redis_client
    self.r = None
    self.sb = io.StringIO()
    self.last_react_time = datetime.min

  def _clean_response(self, text):
    # Regular expression to find '.assistant' and remove everything after
    pattern = r"[.?!]assistant.*"
    cleaned_text = re.sub(pattern, '.', text, flags=re.DOTALL)
    return cleaned_text


  async def write(self, s):
      # Define the unwanted token
      #print(f'PrepostProcess: {s}')
      #s = self._remove_unwanted_fragments(s)
      #unwanted_tokens = ["<|im_end|>", "<|im_start|>"]
      # Remove the unwanted token from 's' if it exists
      #for token in unwanted_tokens:
        #s = s.replace(token, "").rstrip()
      # Write the new content to the buffer
      #print(f'PostPostProcess: {s}')
      print(f"PrepostProcess Response: {s}")
      s = self._clean_response(s)
      self.sb.write(s)
      
      # Split Messages
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

      # Single message
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
          self.r = await self.channel.send(value)


class Discollama:
  def __init__(self, ollama: ollama.AsyncClient, discord, redis, model, activity=None, bot_name=None):
    self.ollama = ollama
    self.discord = discord
    self.redis = redis
    self.model = model
    self.activity_config = activity
    self.bot_name = bot_name
    self.last_react_time = datetime.min

    # register event handlers
    self.discord.event(self.on_ready)
    self.discord.event(self.on_message)

  async def on_ready(self):
      if self.activity_config:
          activity = discord.Activity(name=self.activity_config['name'], type=discord.ActivityType.custom)
          if 'state' in self.activity_config:
              activity.state = self.activity_config['state']
              print(activity.state)
          await self.discord.change_presence(activity=activity)
      
      logging.info(f" {self.bot_name} is ready and online!")

  async def on_message(self, message: discord.message):
    # Start timer
    start_time = perf_counter()  

    # Filter bot messages
    if message.author.bot:
        return
    


    # Channel Type
    channel = message.channel
    if not isinstance(message.channel, discord.DMChannel):
      # Define allowed channel types for processing
      allowed_channel_types = {discord.ChannelType.text, discord.ChannelType.public_thread}

      # Check if the channel type is allowed
      if message.channel.type not in allowed_channel_types:
          return
      
      # Thread
      if message.channel.type == discord.ChannelType.public_thread:
        if not message.channel.owner_id == self.discord.user.id:
          return
        # Handle trespassers
        thread_creator_id = self.redis.get(f"thread:{message.channel.id}:creator")
        if thread_creator_id and message.author.id != int(thread_creator_id):
            await message.delete()
            dm_channel = await message.author.create_dm()
            await dm_channel.send("Please stay in your lane! 😘")
            return
        print(f'Thread Creator id: {thread_creator_id}\n'
              f'Message Author id: {message.author.id}')
      # Text
      elif message.channel.type == discord.ChannelType.text:
        if not self.discord.user.mentioned_in(message):
          return
        if message.mention_everyone:
          return
        # Create Thread
        thread_name = f'{message.author.display_name} x {self.bot_name}'
        channel = await message.channel.create_thread(
           name=thread_name,
           message=message, 
           auto_archive_duration=10080)
        self.redis.set(f"thread:{channel.id}:creator", message.author.id)
        print(f"thread:{channel.id}:creator\n"
              f'author id: {message.author.id}')

    # Check IDs during Maintenance
    maintenance = False
    whitelist = [406159439457157130, 1232999654933925900, 132080494387920896]
    if maintenance and message.author.id not in whitelist:
        await channel.send("\n**We are switching to Llama3! Please stop by later~**")
        return
        
    # Clean @mention
    content = message.content.replace(f'<@{self.discord.user.id}>', f'{self.discord.user.id}>').strip() if self.discord.user.mentioned_in(message) else message.content
    if not content:
      content = 'Hi!' # Fallback

    # Check /[command] 
    if message.content.strip().lower() == '/forget':
      channel_id = message.channel.id if not isinstance(message.channel, discord.DMChannel) else None
      user_id = message.author.id if isinstance(message.channel, discord.DMChannel) else None
      await self.forget_conversation(channel_id=channel_id, user_id=user_id)
      await message.channel.send("\n**Here's to a new us! 🥂**")
      return

    
    context = await self.load(channel_id=channel.id if not isinstance(message.channel, discord.DMChannel) else None, 
                              user_id=message.author.id if isinstance(message.channel, discord.DMChannel) else None)

    r = Response(message, channel, self.redis, self.discord.user.id, self.bot_name)
    
    async with channel.typing():
        # Generate and send response
        async for part in self.generate(content, context):
            # Check task status before cancellation
            logging.info(f"About to send response: {part['response'][:500]}-")
            #print(f"Response Context: {part['context']}")
            await r.write(part['response'])

        await r.write('')

        # Calculate response time
        end_time = perf_counter()
        response_time = end_time - start_time

        # Logging
        log_data = LogData(
            channel_type='DM' if isinstance(message.channel, discord.DMChannel) else 'Public',
            user_channel_id=message.author.id if isinstance(message.channel, discord.DMChannel) else message.channel.id,
            context_length=len(context),
            response_time=response_time,
            prompt_length=len(content),
            response_length=len(part['response'])
        )
        log_performance(log_data)

        # Save redis keys
        await self.save(r.channel.id, message.id, part['context'], message.author.id if isinstance(message.channel, discord.DMChannel) else None)


  async def generate(self, content, context_ids):
      
      # Format prompt
      llama = True
      if llama is True:
        template = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            "{{ .Response }}<|eot_id|>"
        )
      else:
        template = f"system\n{self.bot_name}\nuser\n{content}\nassistant\n"
      #print(f'Template: {template}')
      #print(f'Context: {context_ids}')
      # Feed prompt & context to model
      async def request_with_timeout():
          try:
              response = await self.ollama.generate(model=self.model, prompt=content, system=self.bot_name, context=context_ids, keep_alive=-1, stream=False)
              return response
          except asyncio.TimeoutError:
              logging.warning("Timeout")
              return {"response": "Request timed out.", "context": []}
          except Exception as e:
              logging.exception("Exception in generate:", exc_info=e)
              return {"response": "Error generating response.", "context": []}
      # Timeout
      response = await asyncio.wait_for(request_with_timeout(), timeout=90.0)
      yield response

  async def save(self, channel_id, message_id, ctx: list[int], user_id=None):
      key_prefix = f'discollama:{self.discord.user.id}:dm:' if user_id else f'discollama:{self.discord.user.id}:channel:'
      key = f"{key_prefix}{user_id or channel_id}"
      self.redis.set(key, json.dumps(message_id), ex=60 * 60 * 24 * 7)
      self.redis.set(f'discollama:message:{message_id}', json.dumps(ctx), ex=60 * 60 * 24 * 7)

      # Track message IDs in a set for DM conversations
      if user_id:
          messages_set_key = f'discollama:{self.discord.user.id}:user_messages:{user_id}'
          self.redis.sadd(messages_set_key, message_id)
          # Set auto-expiration
          self.redis.expire(messages_set_key, 60 * 60 * 24 * 7)

  async def load(self, channel_id=None, user_id=None) -> list[int]:
      message_id = None
      key_prefix = f'discollama:{self.discord.user.id}:dm:' if user_id else f'discollama:{self.discord.user.id}:channel:'
      key = f"{key_prefix}{user_id or channel_id}"
      print(f"Loading context for key: {key}")
      message_id = self.redis.get(key)
      ctx = self.redis.get(f'discollama:message:{message_id}')
      print(f'Context: {ctx}')
      return json.loads(ctx) if ctx else []
  
  async def forget_conversation(self, channel_id=None, user_id=None):
      # Determine if the conversation is a DM or belongs to a thread/channel
      key_prefix = f'discollama:{self.discord.user.id}:dm:' if user_id else f'discollama:{self.discord.user.id}:channel:'
      key = f"{key_prefix}{user_id or channel_id}"
      message_ids_set_key = f'discollama:{self.discord.user.id}:user_messages:{user_id or channel_id}'
      
      # Retrieve all message IDs stored in Redis for the user or channel/thread
      message_ids = self.redis.smembers(message_ids_set_key)

      # Delete each message context key stored in Redis
      for msg_id in message_ids:
          self.redis.delete(f'discollama:message:{msg_id}')

      # Delete the set itself and the latest message key
      self.redis.delete(message_ids_set_key)
      self.redis.delete(key)
  
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
