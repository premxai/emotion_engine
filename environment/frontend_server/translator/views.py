"""
Author: Joon Sung Park (joonspk@stanford.edu)
File: views.py
"""
import os
import string
import random
import json
from pathlib import Path
from os import listdir
import os

import datetime
from django.shortcuts import render, redirect, HttpResponseRedirect
from django.http import Http404, HttpResponse, JsonResponse
from global_methods import *

from django.contrib.staticfiles.templatetags.staticfiles import static
from .models import *


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GHOST_TOWN_OUTPUTS = PROJECT_ROOT / "outputs"
GHOST_TOWN_SPRITES = [
  "Isabella_Rodriguez",
  "Klaus_Mueller",
  "Maria_Lopez",
  "Arthur_Burton",
  "Ayesha_Khan",
  "Ryan_Park",
  "Yuriko_Yamamoto",
  "Tamara_Taylor",
  "Carmen_Ortiz",
  "Sam_Moore",
  "Jennifer_Moore",
  "Adam_Smith",
  "Abigail_Chen",
  "Giorgio_Rossi",
  "Carlos_Gomez",
  "Mei_Lin",
  "John_Lin",
  "Eddy_Lin",
  "Tom_Moreno",
  "Jane_Moreno",
  "Wolfgang_Schulz",
  "Rajiv_Patel",
  "Latoya_Williams",
  "Hailey_Johnson",
  "Francisco_Lopez",
]


def _build_persona_entries(persona_names, sprite_aliases=None):
  entries = []
  for idx, persona_name in enumerate(persona_names):
    underscore = persona_name.replace(" ", "_")
    sprite = underscore
    if sprite_aliases:
      sprite = sprite_aliases.get(persona_name, sprite)
    entries += [{
      "original": persona_name,
      "underscore": underscore,
      "initial": persona_name[0] + persona_name.split(" ")[-1][0],
      "sprite_underscore": sprite,
      "profile_underscore": sprite,
    }]
  return entries


def _build_world_config(world_id, character_map=None):
  if world_id == "ghost_town":
    return {
      "world_id": "ghost_town",
      "asset_root": "assets/ghost_town",
      "tilemap_json": "assets/ghost_town/visuals/ghost_town.json",
      "map_name": "ghost_town",
      "supports_ghosts": True,
      "character_map": character_map or {},
      "detail_route": "ghost_town_persona_state",
    }
  return {
    "world_id": "the_ville",
    "asset_root": "assets/the_ville",
    "tilemap_json": "assets/the_ville/visuals/the_ville_jan7.json",
    "map_name": "the_ville",
    "supports_ghosts": False,
    "character_map": character_map or {},
    "detail_route": "replay_persona_state",
  }


def _require_existing_file(file_path, missing_message):
  if not Path(file_path).exists():
    raise Http404(missing_message)
  return file_path


def _prepare_demo_payload(raw_all_movement, step):
  persona_names = []
  persona_names_set = set()
  first_frame = raw_all_movement["0"]
  for name, payload in first_frame.items():
    if name in ["meta", "ghosts"]:
      continue
    if isinstance(payload, dict) and "movement" in payload:
      persona_names += [name]
      persona_names_set.add(name)

  all_movement = dict()
  init_prep = dict()
  for int_key in range(step+1):
    key = str(int_key)
    val = raw_all_movement[key]
    for p in persona_names_set:
      if p in val:
        init_prep[p] = val[p]
    if "ghosts" in val:
      init_prep["ghosts"] = val["ghosts"]
    if "meta" in val:
      init_prep["meta"] = val["meta"]

  persona_init_pos = dict()
  for p in persona_names_set:
    persona_init_pos[p.replace(" ", "_")] = init_prep[p]["movement"]
  all_movement[step] = init_prep

  for int_key in range(step+1, len(raw_all_movement.keys())):
    all_movement[int_key] = raw_all_movement[str(int_key)]

  return persona_names, persona_init_pos, all_movement


def _ghost_town_character_map(persona_names):
  mapping = dict()
  for idx, persona_name in enumerate(persona_names):
    mapping[persona_name] = GHOST_TOWN_SPRITES[idx % len(GHOST_TOWN_SPRITES)]
  return mapping

def landing(request):
  context = {}
  template = "landing/landing.html"
  return render(request, template, context)


def whitepaper(request):
  return render(request, "whitepaper/whitepaper.html", {})


def demo(request, sim_code, step, play_speed="2"): 
  move_file = f"compressed_storage/{sim_code}/master_movement.json"
  meta_file = f"compressed_storage/{sim_code}/meta.json"
  _require_existing_file(meta_file, f"Replay metadata not found for '{sim_code}'.")
  _require_existing_file(move_file, f"Replay movement log not found for '{sim_code}'.")
  step = int(step)
  play_speed_opt = {"1": 1, "2": 2, "3": 4,
                    "4": 8, "5": 16, "6": 32}
  if play_speed not in play_speed_opt: play_speed = 2
  else: play_speed = play_speed_opt[play_speed]

  # Loading the basic meta information about the simulation.
  meta = dict() 
  with open (meta_file) as json_file: 
    meta = json.load(json_file)

  sec_per_step = meta["sec_per_step"]
  start_datetime = datetime.datetime.strptime(meta["start_date"] + " 00:00:00", 
                                              '%B %d, %Y %H:%M:%S')
  for i in range(step): 
    start_datetime += datetime.timedelta(seconds=sec_per_step)
  start_datetime = start_datetime.strftime("%Y-%m-%dT%H:%M:%S")

  # Loading the movement file
  raw_all_movement = dict()
  with open(move_file) as json_file: 
    raw_all_movement = json.load(json_file)
 
  persona_name_list, persona_init_pos, all_movement = _prepare_demo_payload(raw_all_movement, step)
  persona_names = _build_persona_entries(persona_name_list)
  world_config = _build_world_config("the_ville")

  context = {"sim_code": sim_code,
             "step": step,
             "persona_names": persona_names,
             "persona_init_pos": json.dumps(persona_init_pos), 
             "all_movement": json.dumps(all_movement), 
             "start_datetime": start_datetime,
             "sec_per_step": sec_per_step,
             "play_speed": play_speed,
             "world_config": world_config,
             "world_config_json": json.dumps(world_config),
             "mode": "demo"}
  template = "demo/demo.html"

  return render(request, template, context)


def ghost_town_demo(request, sim_code, step, play_speed="2"):
  run_dir = GHOST_TOWN_OUTPUTS / sim_code
  move_file = run_dir / "master_movement.json"
  meta_file = run_dir / "meta.json"
  _require_existing_file(meta_file, f"Ghost town replay metadata not found for '{sim_code}'.")
  _require_existing_file(move_file, f"Ghost town replay movement log not found for '{sim_code}'.")
  step = int(step)
  play_speed_opt = {"1": 1, "2": 2, "3": 4,
                    "4": 8, "5": 16, "6": 32}
  if play_speed not in play_speed_opt:
    play_speed = 2
  else:
    play_speed = play_speed_opt[play_speed]

  with open(meta_file, encoding="utf-8") as json_file:
    meta = json.load(json_file)
  with open(move_file, encoding="utf-8") as json_file:
    raw_all_movement = json.load(json_file)

  persona_name_list, persona_init_pos, all_movement = _prepare_demo_payload(raw_all_movement, step)
  character_map = _ghost_town_character_map(persona_name_list)
  persona_names = _build_persona_entries(persona_name_list, character_map)
  world_config = _build_world_config("ghost_town", character_map)

  context = {
    "sim_code": sim_code,
    "step": step,
    "persona_names": persona_names,
    "persona_init_pos": json.dumps(persona_init_pos),
    "all_movement": json.dumps(all_movement),
    "start_datetime": "2026-01-01T06:00:00",
    "sec_per_step": meta.get("sec_per_step", 300),
    "play_speed": play_speed,
    "world_config": world_config,
    "world_config_json": json.dumps(world_config),
    "mode": "demo",
  }
  template = "demo/demo.html"
  return render(request, template, context)


def UIST_Demo(request): 
  return demo(request, "March20_the_ville_n25_UIST_RUN-step-1-141", 2160, play_speed="3")


def home(request):
  f_curr_sim_code = "temp_storage/curr_sim_code.json"
  f_curr_step = "temp_storage/curr_step.json"

  if not check_if_file_exists(f_curr_step): 
    context = {}
    template = "home/error_start_backend.html"
    return render(request, template, context)

  with open(f_curr_sim_code) as json_file:  
    sim_code = json.load(json_file)["sim_code"]
  
  with open(f_curr_step) as json_file:  
    step = json.load(json_file)["step"]

  os.remove(f_curr_step)

  persona_names = []
  persona_names_set = set()
  for i in find_filenames(f"storage/{sim_code}/personas", ""): 
    x = i.split("/")[-1].strip()
    if x[0] != ".": 
      persona_names += [[x, x.replace(" ", "_")]]
      persona_names_set.add(x)

  persona_init_pos = []
  file_count = []
  for i in find_filenames(f"storage/{sim_code}/environment", ".json"):
    x = i.split("/")[-1].strip()
    if x[0] != ".": 
      file_count += [int(x.split(".")[0])]
  curr_json = f'storage/{sim_code}/environment/{str(max(file_count))}.json'
  with open(curr_json) as json_file:  
    persona_init_pos_dict = json.load(json_file)
    for key, val in persona_init_pos_dict.items(): 
      if key in persona_names_set: 
        persona_init_pos += [[key, val["x"], val["y"]]]

  context = {"sim_code": sim_code,
             "step": step, 
             "persona_names": persona_names,
             "persona_init_pos": persona_init_pos,
             "mode": "simulate"}
  template = "home/home.html"
  return render(request, template, context)


def replay(request, sim_code, step): 
  sim_code = sim_code
  step = int(step)

  persona_names = []
  persona_names_set = set()
  for i in find_filenames(f"storage/{sim_code}/personas", ""): 
    x = i.split("/")[-1].strip()
    if x[0] != ".": 
      persona_names += [[x, x.replace(" ", "_")]]
      persona_names_set.add(x)

  persona_init_pos = []
  file_count = []
  for i in find_filenames(f"storage/{sim_code}/environment", ".json"):
    x = i.split("/")[-1].strip()
    if x[0] != ".": 
      file_count += [int(x.split(".")[0])]
  curr_json = f'storage/{sim_code}/environment/{str(max(file_count))}.json'
  with open(curr_json) as json_file:  
    persona_init_pos_dict = json.load(json_file)
    for key, val in persona_init_pos_dict.items(): 
      if key in persona_names_set: 
        persona_init_pos += [[key, val["x"], val["y"]]]

  context = {"sim_code": sim_code,
             "step": step,
             "persona_names": persona_names,
             "persona_init_pos": persona_init_pos, 
             "mode": "replay"}
  template = "home/home.html"
  return render(request, template, context)


def replay_persona_state(request, sim_code, step, persona_name): 
  sim_code = sim_code
  step = int(step)

  persona_name_underscore = persona_name
  persona_name = " ".join(persona_name.split("_"))
  memory = f"storage/{sim_code}/personas/{persona_name}/bootstrap_memory"
  if not os.path.exists(memory): 
    memory = f"compressed_storage/{sim_code}/personas/{persona_name}/bootstrap_memory"

  with open(memory + "/scratch.json") as json_file:  
    scratch = json.load(json_file)

  with open(memory + "/spatial_memory.json") as json_file:  
    spatial = json.load(json_file)

  with open(memory + "/associative_memory/nodes.json") as json_file:  
    associative = json.load(json_file)

  a_mem_event = []
  a_mem_chat = []
  a_mem_thought = []

  for count in range(len(associative.keys()), 0, -1): 
    node_id = f"node_{str(count)}"
    node_details = associative[node_id]

    if node_details["type"] == "event":
      a_mem_event += [node_details]

    elif node_details["type"] == "chat":
      a_mem_chat += [node_details]

    elif node_details["type"] == "thought":
      a_mem_thought += [node_details]
  
  context = {"sim_code": sim_code,
             "step": step,
             "persona_name": persona_name, 
             "persona_name_underscore": persona_name_underscore, 
             "scratch": scratch,
             "spatial": spatial,
             "a_mem_event": a_mem_event,
             "a_mem_chat": a_mem_chat,
             "a_mem_thought": a_mem_thought}
  template = "persona_state/persona_state.html"
  return render(request, template, context)


def ghost_town_persona_state(request, sim_code, step, persona_name):
  persona_name_underscore = persona_name
  persona_name = " ".join(persona_name.split("_"))
  state_file = GHOST_TOWN_OUTPUTS / sim_code / "personas" / persona_name / "state.json"
  _require_existing_file(
    state_file,
    f"Ghost town persona state not found for '{persona_name}' in '{sim_code}'.",
  )
  with open(state_file, encoding="utf-8") as json_file:
    state = json.load(json_file)

  context = {
    "sim_code": sim_code,
    "step": int(step),
    "persona_name": persona_name,
    "persona_name_underscore": persona_name_underscore,
    "state": state,
  }
  template = "persona_state/ghost_town_state.html"
  return render(request, template, context)


def ghost_town_affect_state(request, sim_code, step):
  """Return affect_timeline slice for a given step as JSON.

  GET /ghost_town/affect_state/<sim_code>/<step>/

  Returns a dict keyed by agent name with their affect vector, latent vector,
  action, and probe_data at the requested step. Falls back to nearest earlier
  step if exact step not present. Returns 404 if affect_timeline.json missing.
  """
  run_dir = GHOST_TOWN_OUTPUTS / sim_code
  affect_file = run_dir / "affect_timeline.json"
  if not affect_file.exists():
    # Try full_replay.json merged file
    full_replay = run_dir / "full_replay.json"
    if full_replay.exists():
      with open(full_replay, encoding="utf-8") as fh:
        full = json.load(fh)
      affect_data = full.get("affect_timeline", {})
    else:
      raise Http404(f"affect_timeline.json not found for '{sim_code}'.")
  else:
    with open(affect_file, encoding="utf-8") as fh:
      affect_data = json.load(fh)

  step_str = str(step)
  step_int = int(step)

  # Find exact step or fall back to nearest earlier step
  if step_str in affect_data:
    payload = affect_data[step_str]
  else:
    available = sorted(int(k) for k in affect_data.keys() if k.isdigit())
    candidates = [s for s in available if s <= step_int]
    if not candidates:
      return JsonResponse({}, status=200)
    payload = affect_data[str(candidates[-1])]

  return JsonResponse({"step": step_int, "agents": payload})


def ghost_town_full_replay(request, sim_code):
  """Serve the merged full_replay.json for a ghost town run.

  GET /ghost_town/full_replay/<sim_code>/
  """
  run_dir = GHOST_TOWN_OUTPUTS / sim_code
  full_replay = run_dir / "full_replay.json"
  if not full_replay.exists():
    raise Http404(f"full_replay.json not found for '{sim_code}'. Run manage.py merge_ghost_town_replay first.")
  with open(full_replay, encoding="utf-8") as fh:
    data = json.load(fh)
  return JsonResponse(data)


def path_tester(request):
  context = {}
  template = "path_tester/path_tester.html"
  return render(request, template, context)


def process_environment(request): 
  """
  <FRONTEND to BACKEND> 
  This sends the frontend visual world information to the backend server. 
  It does this by writing the current environment representation to 
  "storage/environment.json" file. 

  ARGS:
    request: Django request
  RETURNS: 
    HttpResponse: string confirmation message. 
  """
  # f_curr_sim_code = "temp_storage/curr_sim_code.json"
  # with open(f_curr_sim_code) as json_file:  
  #   sim_code = json.load(json_file)["sim_code"]

  data = json.loads(request.body)
  step = data["step"]
  sim_code = data["sim_code"]
  environment = data["environment"]

  with open(f"storage/{sim_code}/environment/{step}.json", "w") as outfile:
    outfile.write(json.dumps(environment, indent=2))

  return HttpResponse("received")


def update_environment(request): 
  """
  <BACKEND to FRONTEND> 
  This sends the backend computation of the persona behavior to the frontend
  visual server. 
  It does this by reading the new movement information from 
  "storage/movement.json" file.

  ARGS:
    request: Django request
  RETURNS: 
    HttpResponse
  """
  # f_curr_sim_code = "temp_storage/curr_sim_code.json"
  # with open(f_curr_sim_code) as json_file:  
  #   sim_code = json.load(json_file)["sim_code"]

  data = json.loads(request.body)
  step = data["step"]
  sim_code = data["sim_code"]

  response_data = {"<step>": -1}
  if (check_if_file_exists(f"storage/{sim_code}/movement/{step}.json")):
    with open(f"storage/{sim_code}/movement/{step}.json") as json_file: 
      response_data = json.load(json_file)
      response_data["<step>"] = step

  return JsonResponse(response_data)


def path_tester_update(request): 
  """
  Processing the path and saving it to path_tester_env.json temp storage for 
  conducting the path tester. 

  ARGS:
    request: Django request
  RETURNS: 
    HttpResponse: string confirmation message. 
  """
  data = json.loads(request.body)
  camera = data["camera"]

  with open(f"temp_storage/path_tester_env.json", "w") as outfile:
    outfile.write(json.dumps(camera, indent=2))

  return HttpResponse("received")









