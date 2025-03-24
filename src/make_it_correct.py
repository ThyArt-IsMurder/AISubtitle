def make_it_correct_openai(message, system_prompt):
    import openai
    openai.api_key = "KEY"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message['content']