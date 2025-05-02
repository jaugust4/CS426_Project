from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd

# This class stores all the stats for a Pokemon
class PokemonStats:
    def __init__(self, hp, attack, defense, sp_attack, sp_defense, speed):
        # Basic stats that every Pokemon has
        self.hp = hp
        self.attack = attack
        self.defense = defense
        self.sp_attack = sp_attack
        self.sp_defense = sp_defense
        self.speed = speed
        # Total of all stats
        self.total_stats = hp + attack + defense + sp_attack + sp_defense + speed

# This class represents a single Pokemon
class Pokemon:
    def __init__(self, name, generation, type1, type2, stats, is_legendary):
        # Basic info about the Pokemon
        self.name = name
        self.generation = generation
        self.type1 = type1
        self.type2 = type2
        # All the Pokemon's stats
        self.stats = stats
        # Whether it's a legendary Pokemon
        self.is_legendary = is_legendary

    # This method helps us create a Pokemon from a row in our data
    @classmethod
    def from_dataframe(cls, row):
        # Get all the stats from the data
        stats = PokemonStats(
            hp=row['hp'],
            attack=row['attack'],
            defense=row['defense'],
            sp_attack=row['sp_attack'],
            sp_defense=row['sp_defense'],
            speed=row['speed']
        )
        # Create a new Pokemon with all the info
        return cls(
            name=row['name'],
            generation=row['generation'],
            type1=row['type1'],
            type2=row['type2'],
            stats=stats,
            is_legendary=row['is_legendary']
        )

# This class helps us analyze all the Pokemon data
class PokemonAnalysis:
    def __init__(self, pokemon_data):
        # Store all the Pokemon
        self.pokemon_data = pokemon_data

    # Count how many Pokemon are in each generation
    def get_generation_stats(self):
        # Make a dictionary to store the counts
        gen_stats = {}
        # Go through each Pokemon
        for pokemon in self.pokemon_data:
            gen = pokemon.generation
            # If we haven't seen this generation before, start counting
            if gen not in gen_stats:
                gen_stats[gen] = {
                    'count': 0,
                    'legendary_count': 0,
                    'total_stats': 0
                }
            # Add to the counts
            gen_stats[gen]['count'] += 1
            gen_stats[gen]['total_stats'] += pokemon.stats.total_stats
            # If it's legendary, add to the legendary count
            if pokemon.is_legendary:
                gen_stats[gen]['legendary_count'] += 1
        return gen_stats

    # Count how many Pokemon have each type
    def get_type_distribution(self):
        # Make a dictionary to store the counts
        type_counts = {}
        # Go through each Pokemon
        for pokemon in self.pokemon_data:
            # Count the first type
            if pokemon.type1 not in type_counts:
                type_counts[pokemon.type1] = 0
            type_counts[pokemon.type1] += 1
            # Count the second type if it exists
            if pokemon.type2:
                if pokemon.type2 not in type_counts:
                    type_counts[pokemon.type2] = 0
                type_counts[pokemon.type2] += 1
        return type_counts