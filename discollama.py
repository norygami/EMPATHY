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
            self.r = None

      # Single message
      else:
          value = self.sb.getvalue().strip()
          if value:  # There's something to send
              await self._send_or_edit(value)
              # Reset the buffer after sending/editing
              self.sb.seek(0, io.SEEK_SET)
              self.sb.truncate()

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
    thread_created = False

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
        thread_created = True
        print(f"thread:{channel.id}:creator\n"
              f'author id: {message.author.id}')

    # Check IDs during Maintenance
    maintenance = False
    whitelist = [406159439457157130, 1232999654933925900, 132080494387920896]
    if maintenance and message.author.id not in whitelist:
        await channel.send("\n**Gated access! Please stop by later~**")
        return
        
    # Clean @mention
    message.content = message.content.replace(f'<@{self.discord.user.id}>', '').strip() if self.discord.user.mentioned_in(message) else message.content
    if not message.content:
      message.content = 'Hi!' # Fallback

    # Check /[command] 
    if message.content.strip().lower() == '/forget':
      channel_id = message.channel.id if not isinstance(message.channel, discord.DMChannel) else None
      user_id = message.author.id if isinstance(message.channel, discord.DMChannel) else None
      await self.forget_conversation(channel_id=channel_id, user_id=user_id)
      await message.channel.send("\n**Here's to a new us! 🥂**")
      return

    # Instantiate Response
    r = Response(message, channel, self.redis, self.discord.user.id, self.bot_name)
    
    # Call Generate
    async with channel.typing():
        # Generate and send response
        response_content, history_length = await self.generate(message, channel.id if thread_created else message.channel.id)
        await r.write(response_content)
        await r.write('')

        # Calculate response time
        end_time = perf_counter()
        response_time = end_time - start_time

        # Logging
        log_data = LogData(
            channel_type='DM' if isinstance(message.channel, discord.DMChannel) else 'Public',
            user_channel_id=message.author.id if isinstance(message.channel, discord.DMChannel) else message.channel.id,
            response_time=response_time,
            response_length=len(response_content),
            history_length=history_length,
        )
        log_performance(log_data)


  # Generate Response
  async def generate(self, message: discord.Message, channel_id):
      # Prepare message history for chat
      history = await self.load_message_history(message.channel.id)

      # System message
      system_message = self.prepare_message('system', f'{self.bot_name}')

      # Check if the last system message differs from the current one
      if not history or (history[-1]['role'] == 'system' and history[-1]['content'] != system_message['content']):
          history.append(system_message)
      
      # User message
      user_content = message.content.strip() if message.content.strip() else "Hello there!"
      user_message = self.prepare_message('user', user_content)
      history.append(user_message)

      # Call chat API
      async def request_with_timeout():
          try:
              response = await self.ollama.chat(
                  model=self.model,
                  messages=history,
                  stream=False
              )
              return response
          except asyncio.TimeoutError:
              logging.warning("Timeout during chat request")
              return {'content': "The response took too long and timed out.", 'role': 'assistant'}
          except Exception as e:
              logging.exception("Exception during chat request:", exc_info=e)
              return {'content': "An error occurred while generating a response.", 'role': 'assistant'}
      
      response = await asyncio.wait_for(request_with_timeout(), timeout=90.0)

        # Check if content is present and non-empty
      if 'message' in response and 'content' in response['message'] and response['message']['content']:
          history.append({
              'role': 'assistant',
              'content': response['message']['content']
          })
          logging.info("Received valid response from API.")
      else:
          # Handle cases where the response is lacking content
          logging.error("Expected 'content' not found in response. Full response: {}".format(response))
          response['message'] = {'content': "Sorry, I couldn't process that request or there was no input.", 'role': 'assistant'}
          history.append(response['message'])

      # Always save the updated message history to the database
      await self.save_message_history(channel_id, history)
      print(f'History after API response: {history}')

      # Return the last message content, typically the assistant's response
      return history[-1]['content'], len(history)
  
  
  def prepare_message(self, role, content):
      return {'role': role, 'content': content}

  def log_history_action(self, action, channel_id, data):
    logging.info(f"{action} history for channel {channel_id}: {data}")
  
  async def load_message_history(self, channel_id) -> list:
      key = f"history:{channel_id}"
      history_json = self.redis.get(key)
      self.log_history_action("Loaded", channel_id, history_json)
      return json.loads(history_json) if history_json else []

  async def save_message_history(self, channel_id, history):
      key = f"history:{channel_id}"
      history_json = json.dumps(history)
      self.redis.set(key, history_json, ex=60 * 60 * 24 * 7)  # 7 days expiration for simplicity
      logging.info(f"Saved history for channel {channel_id}: {history_json}")

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
  
  
  # Setup
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
