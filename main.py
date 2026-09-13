import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    information = """Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is a businessman and former public official who is the CEO and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and briefly became the only trillionaire (in terms of US dollars) in June 2026; as of September 2026, Forbes estimates his net worth to be US$908 billion.

        Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002."
    """
    summary_template = """Given the information {information}, about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm  = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens=500
    )

    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information":information})
    print(response.content)

if __name__ == "__main__":
    main()
