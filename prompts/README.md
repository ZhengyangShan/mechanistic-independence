# Prompt templates

Drop the five prompt-template files here (or point ``$MMI_DATA_ROOT`` at a
directory that already contains them):

```
race_name_prompts.txt
gender_name_prompts.txt
race_profession_prompts.txt
gender_profession_prompts.txt
education_profession_prompts.txt
```

Each file contains many prompt blocks separated by a single line ``---``. A
block is plain text that mentions the items the model should label, e.g.

```
Here is a list of words. The names are: Jamal, Alice, Wei, Maria.
For each name, predict whether the person is Black, White, Asian, or
Hispanic. Format each line as: Name - <Label>. Respond only with the list.
```

Items are extracted with the regex in ``src/prompts.py``; ``The names are:
…``, ``The professions are: …``, and ``The words are: …`` are all recognized.

The repository ships **without** the templates used in the paper. Researchers
who want to reproduce the paper's exact runs should obtain or recreate them
from the description in §4.1.
