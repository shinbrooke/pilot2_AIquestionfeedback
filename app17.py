import streamlit as st
import time
import pandas as pd
import json
import os
import random
from datetime import datetime
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# langchain related imports
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException
from pydantic import BaseModel, Field

# streamlit cache related import
from functools import lru_cache

# Pydantic models for structured output
class BloomClassification(BaseModel):
    bloom_level: str = Field(description="The Bloom's taxonomy level: 기억, 이해, 적용, 분석, 평가, or 창조")

class QuestionSuggestion(BaseModel):
    suggested_question: str = Field(description="A single suggested question in Korean ending with a question mark")

# Import paragraphs from config file
try:
    from paragraphs_config_revised import get_paragraphs
except ImportError:
    # Fallback if config file doesn't exist
    def get_paragraphs(count=45):
        return [f"Sample paragraph {i+1}" for i in range(count)]

# Load environment variables (for OpenAI API key)
load_dotenv()

# Check if OpenAI API key is available
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: No OpenAI API key found in environment variables.")
    print("Please set OPENAI_API_KEY in your .env file or environment variables.")
else:
    print("OpenAI API key loaded successfully.")

# Few-shot examples for Bloom's taxonomy classification
BLOOM_CLASSIFICATION_EXAMPLES = [
    {
        "paragraph": "8세기 후반 바그다드에는 '지혜의 집(Bayt al-Hikma)'이라는 지식 집약 기관이 설립되어, 고대 그리스의 철학과 자연과학 문헌을 아랍어로 번역하는 대규모 작업이 이루어졌다. 이 번역은 단순히 언어를 바꾸는 것이 아니라, 플라톤, 아리스토텔레스, 히포크라테스 등의 사상을 해석하고 보완하며 새로운 학문 체계를 세우는 과정이었다.",
        "question": "지혜의 집은 언제 설립되었을까?",
        "bloom_level": "기억"
    },
    {
        "paragraph": "현대 도시계획에서 녹지 공간의 중요성이 대두되고 있다. 녹지는 대기 정화, 온도 조절, 시민의 정신 건강 향상 등 다양한 기능을 수행한다. 특히 팬데믹 이후 야외 활동 공간에 대한 수요가 급증하면서, 도시 내 공원과 정원의 역할이 재조명받고 있다.",
        "question": "녹지 공간이 시민들에게 제공하는 주요 혜택들은 무엇일까?",
        "bloom_level": "이해"
    },
    {
        "paragraph": "예측 처리 이론(predictive processing theory)은 인간의 뇌가 외부 자극을 받아들이기만 하는 수동적 기관이 아니라, 끊임없이 미래의 감각 정보를 예측하고 그 예측이 실제 감각 정보와 얼마나 일치하는지를 비교하면서 작동한다고 설명한다. 즉, 뇌는 예상과 다른 정보가 들어올 때 그 오류를 수정해나가는 방식으로 작동한다. 예측이 잘 맞으면 뇌의 에너지 사용은 줄어들고, 오류가 있으면 더 많은 자원이 동원되어 환경에 대한 새로운 모델을 학습하게 된다. 이 이론은 '정보가 입력되고 처리된다'는 고전적 인지 이론과 달리, 뇌가 능동적으로 세계를 구성한다는 관점의 전환을 보여준다. 또한, 주의, 정서, 자아감 형성과 같은 복잡한 심리 현상까지 설명할 수 있는 인지 모델을 제공한다.",
        "question": "뇌가 예측을 통해 감각 정보를 처리한다는 주장을 시각 경험을 예로 들면 어떻게 설명할 수 있을까?",
        "bloom_level": "적용"
    },
    {
        "paragraph": "최근 연구에 따르면, 나뭇잎소리, 새소리, 물소리와 같은 자연의 소리를 듣는 것만으로도 스트레스 수치가 낮아지고 집중력이 향상될 수 있다고 한다. 실험 참가자들이 인공적인 도시 소음과 자연의 소리를 각각 들었을 때, 자연의 소리를 들은 그룹은 심박수와 코르티솔 수치가 낮아졌고, 주의 전환 속도와 기억력에서 더 높은 성과를 보였다. 연구자들은 자연의 소리가 뇌의 주의 회복 시스템을 자극해, 과도한 정보 처리에서 벗어나게 돕는다고 설명한다. 이는 단순히 조용한 환경이 주는 효과가 아니라, 자연 특유의 리듬과 패턴이 신경계에 긍정적 영향을 주기 때문으로 보인다. 이러한 연구 결과는 일상생활에서 자연 소리를 의도적으로 접하는 것이 정신 건강 증진에 실질적인 도움이 될 수 있음을 시사한다.",
        "question": "자연의 소리와 인공적인 소음 간에 존재하는 차이 중 어떤 요소가 스트레스 수치 또는 집중력 등과 관련이 있는 것일까?",
        "bloom_level": "분석"
    },
    {
        "paragraph": "텍스트 외 존재론(Ontology Outside of Text)은 해체주의 이후의 철학과 문학이론에서 등장한 개념으로, 언어 바깥의 세계와 경험을 이해하려는 시도를 말한다. 기존의 문학 이론은 주로 언어, 기호, 담론을 통해 인간의 현실을 해석했지만, 이 이론은 그것만으로는 설명되지 않는 실제 삶의 층위를 강조한다. 데리다의 해체론이 모든 의미는 언어 안에서 차이와 지연으로 구성된다고 본 반면, 텍스트 외 존재론은 언어로 포착되지 않는 감각, 몸, 침묵 같은 요소들에도 주목한다. 이 관점은 예술이나 문학에서 말로 설명되지 않는 감정이나 경험을 이해하는 데 도움을 준다. 결국 텍스트 외 존재론은 언어 중심의 사고에서 벗어나 인간 존재에 대한 보다 폭넓은 이해를 추구한다.",
        "question": "언어 중심의 해석 방법과 비교할 때 이론적으로 어떤 한계가 있을까?",
        "bloom_level": "평가"
    },
    {
        "paragraph": "기억의 장소(sites of memory)는 공동체의 역사적 경험이나 정체성이 구체적인 지리적 공간에 응축되어 저장된 장소를 의미하며, 예술은 이를 서사적으로 재구성하는 중요한 매체로 작동한다. 특히 역사적 트라우마와 같은 복잡한 주제들을 다루는 예술 작품은 과거의 사건을 현재의 감각과 윤리 속으로 불러오는 적극적인 재구성 작업을 수행한다. 예술적 재현은 공식 기록으로 남지 않은 기억의 공백을 채우고, 소외된 기억들을 복원함으로써 개인의 기억과 집단적 기억 사이의 경계를 흐리게 만든다. 이러한 작업은 관람자가 기억의 참여자이자 해석자로 전환되도록 유도한다. 이처럼 예술은 단순한 표현 수단을 넘어, 기억의 정치성과 윤리성, 사회적 기억의 구성 방식을 비판적으로 탐구하는 도구로 기능한다.",
        "question": "기억의 예술적 재현은 역사적 사실 검증과 어떤 측면에서 긴장 관계를 맺을 수 있을까?",
        "bloom_level": "창조"
    },
    {
        "paragraph": "코그니타리아트는 후기 자본주의 체제에서 인지, 정동, 창의성을 중심 자산으로 동원당하는 신(新)노동계급을 지칭하는 개념이다. 이들은 비물질적 노동의 수행 주체로서, 디지털 네트워크에 매개된 작업 환경 속에서 자기표현과 성과 창출의 무한한 자기책임성을 강요받는다. '자유로운 창조자'라는 표상 이면에는 플랫폼 자본주의가 조장한 노동의 유연화와 생계의 불확실성이 구조적으로 깊게 내재되어 있다. 이로써 코그니타리아트는 근대적 프롤레타리아트와 달리, 신자유주의적 자아 기술을 통해 자기 착취에 스스로를 능동적으로 동원하게 되는 존재로 전락한다. 이 개념은 노동의 본질이 물질에서 정보로 이행함에 따라, 권력과 저항의 지형 또한 근본적으로 재편되고 있음을 날카롭게 시사한다.",
        "question": "코그니타리아트가 겪는 자기 착취 문제를 줄이기 위한 새로운 노동 구조나 제도는 어떻게 설계할 수 있을까?",
        "bloom_level": "창조"
    }
]

# Few-shot examples for related question generation
RELATED_QUESTION_EXAMPLES = [
    {
        "paragraph": "'미토포에시스(Mythopoeia)'는 단순히 기존 신화를 분석하는 데 그치지 않고, 작가가 자신만의 신화 체계를 창조하는 창작 행위를 의미한다. 이 개념은 특히 C.S. 루이스의 『나니아 연대기』에서 잘 드러나며, 그는 고유한 존재들, 종교적 상징, 윤리적 질서를 나니아라는 유기적 세계로 구성하였다.. 그의 작업은 단순한 판타지를 넘어서, 선과 악의 대립 같은 신화적 주제를 통해 인간 존재의 의미를 탐구하려는 시도였다. 이러한 미토포에시스는 고대 문명처럼 상징과 서사를 통해 세계를 설명하려는 인간의 본능과도 관련이 깊다. 현대의 문학 작품, 판타지 게임, 영화 시나리오에서도 미토포에시스는 중요한 내러티브 기법으로 활용되며, 이는 신화가 여전히 살아 있는 사유 방식임을 보여준다.",
        "user_question": "루이스는 왜 자신만의 신화를 창조하고자 했을까?",
        "suggested_question": "루이스의 신화 창작 방식에서 영감을 받아, 오늘날 우리 사회를 반영한 새로운 신화 체계를 구상하려면 어떤 세계관과 주제를 탐색해볼 수 있을까?"
    },
    {
        "paragraph": "리퀴드 모더니티(liquid modernity)는 지그문트 바우만이 제시한 개념으로, 현대 사회의 유동성과 불확실성을 설명한다. 고체적 근대가 견고한 제도와 안정된 정체성을 기반으로 했다면, 리퀴드 모더니티는 관계, 노동, 소비 방식 모두가 유동적이며 일시적인 특성을 띤다. 리퀴드 모더니티를 보이는 사회는 개인에게 유연성과 선택의 자유를 제공하지만, 동시에 지속적인 자기 재구성과 정체성의 불안을 초래한다. 이 개념은 글로벌화, 디지털화, 개인화가 지배적인 시대에서 사회적 연대와 소속감의 해체를 분석하는 데 효과적으로 활용될 수 있다. 따라서 리퀴드 모더니티는 현대인의 삶의 조건을 해석하고 사회 정책의 방향을 모색하는 데 중요한 이론적 틀을 제공한다.",
        "user_question": "고체 근대와 리퀴드 모더니티의 차이는 무엇일까?",
        "suggested_question": "고체 근대와 리퀴드 모더니티의 정체성 형성 방식 차이를 바탕으로, 디지털 플랫폼에서의 자기 표현은 어떤 새로운 윤리적 쟁점을 낳을 수 있을까?"
    },
    {
        "paragraph": "예측 처리 이론(predictive processing theory)은 인간의 뇌가 외부 자극을 받아들이기만 하는 수동적 기관이 아니라, 끊임없이 미래의 감각 정보를 예측하고 그 예측이 실제 감각 정보와 얼마나 일치하는지를 비교하면서 작동한다고 설명한다. 즉, 뇌는 예상과 다른 정보가 들어올 때 그 오류를 수정해나가는 방식으로 작동한다. 예측이 잘 맞으면 뇌의 에너지 사용은 줄어들고, 오류가 있으면 더 많은 자원이 동원되어 환경에 대한 새로운 모델을 학습하게 된다. 이 이론은 '정보가 입력되고 처리된다'는 고전적 인지 이론과 달리, 뇌가 능동적으로 세계를 구성한다는 관점의 전환을 보여준다. 또한, 주의, 정서, 자아감 형성과 같은 복잡한 심리 현상까지 설명할 수 있는 인지 모델을 제공한다.",
        "user_question": "뇌가 예측을 통해 감각 정보를 처리한다는 주장을 시각 경험을 예로 들면 어떻게 설명할 수 있을까?",
        "suggested_question": "시각 경험을 예로 들 때, 정서 상태가 뇌의 예측에 어떤 영향을 주는지를 알아보는 실험을 기획한다면 어떤 요소를 포함해야 할까?"
    },
    {
        "paragraph": "유전자 가위 기술, 예를 들어 CRISPR-Cas9 시스템은 특정 DNA 염기서열을 정밀하게 절단하고 편집할 수 있게 하여 생명과학 연구에 혁신을 가져왔다. 이 기술은 바이러스에 대항하는 박테리아의 면역 체계에서 유래되었으며, 연구자들은 이를 활용해 유전병 치료, 작물 개량, 생물 다양성 보존 등 다양한 분야에 적용하고 있다. 유전자 가위 기술은 기존의 유전자 조작 기술보다 훨씬 간편하고 저렴하며, 편집의 정밀도가 높아 다양한 생물학적 연구에 핵심 도구로 자리잡고 있다. 하지만 생식세포 유전자 편집과 관련된 윤리적 논쟁, 생태계에 미치는 영향 등에 대해서는 여전히 활발한 논의가 진행 중이다. CRISPR는 생명과 기술, 윤리가 얽힌 복합적 문제들을 우리에게 제기한다.",
        "user_question": "생태계의 균형을 고려하여 CRISPR 기술의 응용을 조절할 수 있는 정책 방안을 고안해본다면 어떤 요소를 고려해야 할까?",
        "suggested_question": "유전자 편집 기술이 생태계에 미치는 영향을 사전에 평가하기 위한 과학적 기준이나 윤리적 기준은 어떤 방식으로 마련될 수 있을까?"
    },
    {
        "paragraph": "자기치유 콘크리트(self-healing concrete)는 콘크리트 구조물에 균열이 발생하더라도 내부의 복원 메커니즘이 작동하여 스스로 파손 부위를 복구할 수 있도록 설계된 지능형 건축 자재다. 이 기술은 박테리아가 석회석을 생성하거나, 고분자 캡슐이 외부 자극에 반응해 복합 물질을 분출하는 등의 원리를 활용하여, 수분과 공기 침투를 막고 구조적 안정성을 연장시키는 방식으로 작동한다. 자기치유 콘크리트는 유지보수 주기를 줄이고 인프라의 전체 수명을 늘리는 데 기여하지만, 초기 제조 비용 증가, 성능의 일관성 확보 문제 등의 한계 역시 존재한다. 특히 극한 온도와 습도, 반복 진동 등 특수한 조건에서도 일관된 치유 성능을 발휘할 때, 지속가능한 도시 인프라를 구현하는 데 기여할 수 있다.",
        "user_question": "자기치유 기술을 다리나 터널 등에 적용하려면 어떤 조건을 고려해야 할까?",
        "suggested_question": "지진, 중차량 통행, 습기 변화가 잦은 지역의 다리에 적용할 자기치유 구조 시스템을 창안해본다면 어떤 방식으로 치유 메커니즘을 조정해야 할까?"
    },
    {
        "paragraph": "기억의 장소(sites of memory)는 공동체의 역사적 경험이나 정체성이 구체적인 지리적 공간에 응축되어 저장된 장소를 의미하며, 예술은 이를 서사적으로 재구성하는 중요한 매체로 작동한다. 특히 역사적 트라우마와 같은 복잡한 주제들을 다루는 예술 작품은 과거의 사건을 현재의 감각과 윤리 속으로 불러오는 적극적인 재구성 작업을 수행한다. 예술적 재현은 공식 기록으로 남지 않은 기억의 공백을 채우고, 소외된 기억들을 복원함으로써 개인의 기억과 집단적 기억 사이의 경계를 흐리게 만든다. 이러한 작업은 관람자가 기억의 참여자이자 해석자로 전환되도록 유도한다. 이처럼 예술은 단순한 표현 수단을 넘어, 기억의 정치성과 윤리성, 사회적 기억의 구성 방식을 비판적으로 탐구하는 도구로 기능한다.",
        "user_question": "기억의 예술적 재현은 역사적 사실 검증과 어떤 측면에서 긴장 관계를 맺을 수 있을까?",
        "suggested_question": "공식 역사와 충돌하는 사적인 기억을 바탕으로 다큐멘터리 연극이나 설치 작품을 구성할 때, 사실성과 허구성을 어떻게 조화시킬 수 있을까?"
    }
]

# Few-shot examples for unrelated question generation
UNRELATED_QUESTION_EXAMPLES = [
    {
        "paragraph": "'미토포에시스(Mythopoeia)'는 단순히 기존 신화를 분석하는 데 그치지 않고, 작가가 자신만의 신화 체계를 창조하는 창작 행위를 의미한다. 이 개념은 특히 C.S. 루이스의 『나니아 연대기』에서 잘 드러나며, 그는 고유한 존재들, 종교적 상징, 윤리적 질서를 나니아라는 유기적 세계로 구성하였다.. 그의 작업은 단순한 판타지를 넘어서, 선과 악의 대립 같은 신화적 주제를 통해 인간 존재의 의미를 탐구하려는 시도였다. 이러한 미토포에시스는 고대 문명처럼 상징과 서사를 통해 세계를 설명하려는 인간의 본능과도 관련이 깊다. 현대의 문학 작품, 판타지 게임, 영화 시나리오에서도 미토포에시스는 중요한 내러티브 기법으로 활용되며, 이는 신화가 여전히 살아 있는 사유 방식임을 보여준다.",
        "user_question": "루이스는 왜 자신만의 신화를 창조하고자 했을까?",
        "suggested_question": "고대 문명이 신화를 통해 세계를 설명했던 방식은 현대 사회의 어떤 문제들을 새로운 서사로 다시 말하는 데 어떻게 활용될 수 있을까?"
    },
    {
        "paragraph": "리퀴드 모더니티(liquid modernity)는 지그문트 바우만이 제시한 개념으로, 현대 사회의 유동성과 불확실성을 설명한다. 고체적 근대가 견고한 제도와 안정된 정체성을 기반으로 했다면, 리퀴드 모더니티는 관계, 노동, 소비 방식 모두가 유동적이며 일시적인 특성을 띤다. 리퀴드 모더니티를 보이는 사회는 개인에게 유연성과 선택의 자유를 제공하지만, 동시에 지속적인 자기 재구성과 정체성의 불안을 초래한다. 이 개념은 글로벌화, 디지털화, 개인화가 지배적인 시대에서 사회적 연대와 소속감의 해체를 분석하는 데 효과적으로 활용될 수 있다. 따라서 리퀴드 모더니티는 현대인의 삶의 조건을 해석하고 사회 정책의 방향을 모색하는 데 중요한 이론적 틀을 제공한다.",
        "user_question": "고체 근대와 리퀴드 모더니티의 차이는 무엇일까?",
        "suggested_question": "리퀴드 모더니티가 지배하는 사회에서 '소속감'의 개념을 새롭게 정의하고 이를 측정하는 방법을 고안한다면 어떤 기준이 필요할까?"
    },
    {
        "paragraph": "예측 처리 이론(predictive processing theory)은 인간의 뇌가 외부 자극을 받아들이기만 하는 수동적 기관이 아니라, 끊임없이 미래의 감각 정보를 예측하고 그 예측이 실제 감각 정보와 얼마나 일치하는지를 비교하면서 작동한다고 설명한다. 즉, 뇌는 예상과 다른 정보가 들어올 때 그 오류를 수정해나가는 방식으로 작동한다. 예측이 잘 맞으면 뇌의 에너지 사용은 줄어들고, 오류가 있으면 더 많은 자원이 동원되어 환경에 대한 새로운 모델을 학습하게 된다. 이 이론은 '정보가 입력되고 처리된다'는 고전적 인지 이론과 달리, 뇌가 능동적으로 세계를 구성한다는 관점의 전환을 보여준다. 또한, 주의, 정서, 자아감 형성과 같은 복잡한 심리 현상까지 설명할 수 있는 인지 모델을 제공한다.",
        "user_question": "뇌가 예측을 통해 감각 정보를 처리한다는 주장을 시각 경험을 예로 들면 어떻게 설명할 수 있을까?",
        "suggested_question": "뇌가 반복적으로 예측에 실패할 때, 외부 세계에 대한 인식은 어떻게 변할 수 있는지 시뮬레이션 실험을 설계해볼 수 있을까?"
    },
    {
        "paragraph": "유전자 가위 기술, 예를 들어 CRISPR-Cas9 시스템은 특정 DNA 염기서열을 정밀하게 절단하고 편집할 수 있게 하여 생명과학 연구에 혁신을 가져왔다. 이 기술은 바이러스에 대항하는 박테리아의 면역 체계에서 유래되었으며, 연구자들은 이를 활용해 유전병 치료, 작물 개량, 생물 다양성 보존 등 다양한 분야에 적용하고 있다. 유전자 가위 기술은 기존의 유전자 조작 기술보다 훨씬 간편하고 저렴하며, 편집의 정밀도가 높아 다양한 생물학적 연구에 핵심 도구로 자리잡고 있다. 하지만 생식세포 유전자 편집과 관련된 윤리적 논쟁, 생태계에 미치는 영향 등에 대해서는 여전히 활발한 논의가 진행 중이다. CRISPR는 생명과 기술, 윤리가 얽힌 복합적 문제들을 우리에게 제기한다.",
        "user_question": "생태계의 균형을 고려하여 CRISPR 기술의 응용을 조절할 수 있는 정책 방안을 고안해본다면 어떤 요소를 고려해야 할까?",
        "suggested_question": "다양한 유전자 가위 기술 중에서 특정 목적(예: 작물 개량 vs. 유전병 치료)에 더 적합한 기술은 어떤 기준으로 선택할 수 있을까?"
    },
    {
        "paragraph": "자기치유 콘크리트(self-healing concrete)는 콘크리트 구조물에 균열이 발생하더라도 내부의 복원 메커니즘이 작동하여 스스로 파손 부위를 복구할 수 있도록 설계된 지능형 건축 자재다. 이 기술은 박테리아가 석회석을 생성하거나, 고분자 캡슐이 외부 자극에 반응해 복합 물질을 분출하는 등의 원리를 활용하여, 수분과 공기 침투를 막고 구조적 안정성을 연장시키는 방식으로 작동한다. 자기치유 콘크리트는 유지보수 주기를 줄이고 인프라의 전체 수명을 늘리는 데 기여하지만, 초기 제조 비용 증가, 성능의 일관성 확보 문제 등의 한계 역시 존재한다. 특히 극한 온도와 습도, 반복 진동 등 특수한 조건에서도 일관된 치유 성능을 발휘할 때, 지속가능한 도시 인프라를 구현하는 데 기여할 수 있다.",
        "user_question": "자기치유 기술을 다리나 터널 등에 적용하려면 어떤 조건을 고려해야 할까?",
        "suggested_question": "수분과 공기의 침투를 최소화할 수 있는 새로운 형태의 자기치유 콘크리트를 설계한다면, 어떤 복합 재료와 메커니즘을 조합할 수 있을까?"
    },
    {
        "paragraph": "기억의 장소(sites of memory)는 공동체의 역사적 경험이나 정체성이 구체적인 지리적 공간에 응축되어 저장된 장소를 의미하며, 예술은 이를 서사적으로 재구성하는 중요한 매체로 작동한다. 특히 역사적 트라우마와 같은 복잡한 주제들을 다루는 예술 작품은 과거의 사건을 현재의 감각과 윤리 속으로 불러오는 적극적인 재구성 작업을 수행한다. 예술적 재현은 공식 기록으로 남지 않은 기억의 공백을 채우고, 소외된 기억들을 복원함으로써 개인의 기억과 집단적 기억 사이의 경계를 흐리게 만든다. 이러한 작업은 관람자가 기억의 참여자이자 해석자로 전환되도록 유도한다. 이처럼 예술은 단순한 표현 수단을 넘어, 기억의 정치성과 윤리성, 사회적 기억의 구성 방식을 비판적으로 탐구하는 도구로 기능한다.",
        "user_question": "기억의 예술적 재현은 역사적 사실 검증과 어떤 측면에서 긴장 관계를 맺을 수 있을까?",
        "suggested_question": "'기억의 장소' 개념을 물리적 장소가 아닌 '심리적 상태'로 해석한다면, 예술 작품은 어떤 식으로 트라우마를 재구성할 수 있을까?"
    }
]

# Set this to False to disable parallel port for initial testing
USE_PARALLEL_PORT = False  # Change to True when you're ready to test with actual hardware

if USE_PARALLEL_PORT:
    try:
        from psychopy import parallel
        # Check if ParallelPort exists and is callable
        if hasattr(parallel, 'ParallelPort') and callable(parallel.ParallelPort):
            port = parallel.ParallelPort(address=0x378)
            PARALLEL_PORT_AVAILABLE = True
            print("Parallel port initialized successfully")
        else:
            print("ParallelPort class not available in psychopy.parallel")
            PARALLEL_PORT_AVAILABLE = False
            port = None
    except Exception as e:
        print(f"Parallel port initialization failed: {e}")
        PARALLEL_PORT_AVAILABLE = False
        port = None
else:
    PARALLEL_PORT_AVAILABLE = False
    port = None
    print("Parallel port disabled for testing")

# Event marker values (adjust as needed)
MARKERS = {
    "baseline_start": 20,
    "baseline_end": 21,
    "paragraph_start": 1,
    "paragraph_end": 2,
    "question_input_start": 5,
    "question_input_end": 6,
    "feedback_start": 7,
    "feedback_end": 8,
    "survey_start": 9,
    "survey_end": 10,
    "edit_start": 11,
    "edit_end": 12,
    "edit_textarea_focus": 13  # Add this for edit textarea focus tracking
}

# Function to send marker through parallel port
def send_marker(marker_type):
    if PARALLEL_PORT_AVAILABLE and port is not None:
        try:
            port.setData(MARKERS[marker_type])
            time.sleep(0.05)  # Brief pulse
            port.setData(0)  # Reset
        except Exception as e:
            st.error(f"Error sending marker: {e}")
    else:
        # When parallel port is disabled, just log the marker event
        if USE_PARALLEL_PORT:
            # If we tried to use parallel port but it failed
            print(f"Cannot send marker '{marker_type}': Parallel port not available")
        else:
            # If parallel port is intentionally disabled for testing
            print(f"[TEST MODE] Would send marker: {marker_type} (value: {MARKERS.get(marker_type, 'NA')})")
    
    # Log the marker event regardless of parallel port availability
    log_event(f"MARKER: {marker_type}")

# Function to log events
def log_event_batched(event_description, data=None):
    """Optimized logging with batching for non-critical events"""
    if hasattr(st.session_state, 'logger'):
        st.session_state.logger.add_event(event_description, data)
    else:
        log_event(event_description, data)  # Fallback

def log_event(event_description, data=None):
    if 'event_log' not in st.session_state:
        st.session_state.event_log = []
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    # Determine context based on practice mode
    context = "baseline" if st.session_state.get('baseline_mode', False) else ("practice" if st.session_state.get('practice_mode', False) else "main")
    
    log_entry = {
        "timestamp": timestamp,
        "iteration": st.session_state.iteration,
        "stage": st.session_state.stage,
        "event": event_description,
        "context": context
    }
    
    if data:
        log_entry["data"] = data
        
    st.session_state.event_log.append(log_entry)

# Function to save logs
def save_logs():
    if 'event_log' in st.session_state and st.session_state.event_log:
        # Create directory if it doesn't exist
        if not os.path.exists("logs"):
            os.makedirs("logs")
            
        participant_id = st.session_state.get("participant_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/participant_{participant_id}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(st.session_state.event_log, f, indent=2)
        
        # Also save responses data for easy analysis
        responses_filename = f"logs/responses_{participant_id}_{timestamp}.csv"
        if 'responses' in st.session_state and st.session_state.responses:
            df = pd.DataFrame(st.session_state.responses)
            df.to_csv(responses_filename, index=False)
        
        return filename, responses_filename
    return None, None

# Function to get current CSV data for download
def get_current_csv_data():
    if 'responses' in st.session_state and st.session_state.responses:
        df = pd.DataFrame(st.session_state.responses)
        return df.to_csv(index=False)
    return ""

# Get practice CSV data
def get_practice_csv_data():
    if 'practice_responses' in st.session_state and st.session_state.practice_responses:
        df = pd.DataFrame(st.session_state.practice_responses)
        return df.to_csv(index=False)
    return ""

# Define paragraph categories and indices
# Note: Index 17 is 자연과학, Index 18 is 사회과학 (correction from original ranges)
GENRE_RANGES = {
    "인문학": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "사회과학": [9, 10, 11, 12, 13, 14, 15, 16, 18],
    "자연과학": [17, 19, 20, 21, 22, 23, 24, 25, 26],
    "공학": [27, 28, 29, 30, 31, 32, 33, 34, 35],
    "예체능": [36, 37, 38, 39, 40, 41, 42, 43, 44]
}

# Define excluded and practice paragraph indices
EXCLUDED_INDICES = [3, 4, 7, 9, 11, 12, 22, 24, 25, 27, 29, 31, 37, 38, 40]
PRACTICE_INDICES = [4, 27]  # One from 인문학 (4), one from 공학 (27)

def get_experiment_paragraph_indices():
    """Get indices for the main experiment, excluding specified indices"""
    all_indices = list(range(45))
    experiment_indices = []
    
    for i in all_indices:
        if i not in EXCLUDED_INDICES and i not in PRACTICE_INDICES:
            experiment_indices.append(i)
    
    return experiment_indices

def get_experiment_paragraphs():
    """Get paragraphs for the main experiment using explicit indices"""
    experiment_indices = get_experiment_paragraph_indices()
    all_paragraphs = get_paragraphs(45)
    
    experiment_paragraphs = []
    for idx in experiment_indices:
        experiment_paragraphs.append({
            'index': idx,
            'content': all_paragraphs[idx],
            'genre': get_genre_for_index(idx)
        })
    
    return experiment_paragraphs

def get_practice_paragraphs():
    """Get paragraphs for practice session using explicit indices"""
    all_paragraphs = get_paragraphs(45)
    practice_paragraphs = []
    
    for idx in PRACTICE_INDICES:
        practice_paragraphs.append({
            'index': idx,
            'content': all_paragraphs[idx],
            'genre': get_genre_for_index(idx)
        })
    
    return practice_paragraphs

def get_genre_for_index(index):
    """Return the genre for a given paragraph index"""
    if 0 <= index <= 8:
        return "인문학"
    elif (9 <= index <= 16) or index == 18:
        return "사회과학"
    elif index == 17 or (19 <= index <= 26):
        return "자연과학"
    elif 27 <= index <= 35:
        return "공학"
    elif 36 <= index <= 44:
        return "예체능"
    else:
        return "unknown"

def create_balanced_condition_assignment(participant_id, experiment_paragraphs):
    """
    Create balanced condition assignment ensuring:
    1. Equal distribution within each genre
    2. Equal total distribution across conditions
    3. Randomized assignment per participant
    4. Proper counterbalancing across participants
    """
    # Use participant ID to create a deterministic seed
    seed = hash(participant_id) % (2**32)
    random.seed(seed)
    
    # Group paragraphs by genre
    genre_paragraphs = {}
    for para in experiment_paragraphs:
        genre = para['genre']
        if genre not in genre_paragraphs:
            genre_paragraphs[genre] = []
        genre_paragraphs[genre].append(para)
    
    condition_mapping = {}
    
    # For each genre, assign conditions with balanced distribution
    for genre, paragraphs in genre_paragraphs.items():
        indices = [p['index'] for p in paragraphs]
        
        # Create balanced assignment within genre
        # If odd number, the extra one goes to a random condition
        half = len(indices) // 2
        
        # Shuffle indices for this genre
        random.shuffle(indices)
        
        # Assign first half to 'related', second half to 'unrelated'
        for i, idx in enumerate(indices):
            if i < half:
                condition_mapping[idx] = "related"
            elif i < 2 * half:
                condition_mapping[idx] = "unrelated"
            else:
                # Handle odd numbers by randomly assigning the extra
                condition_mapping[idx] = random.choice(["related", "unrelated"])
    
    return condition_mapping

def create_practice_condition_assignment(participant_id):
    """
    Create balanced condition assignment for practice session.
    Ensures one related and one unrelated feedback.
    """
    # Use participant ID to create a deterministic seed
    seed = hash(participant_id) % (2**32)
    random.seed(seed)
    
    # Create balanced assignment: one related, one unrelated
    conditions = ["related", "unrelated"]
    random.shuffle(conditions)
    
    # Map to practice iteration indices (0, 1)
    practice_condition_mapping = {}
    for i, condition in enumerate(conditions):
        practice_condition_mapping[i] = condition
    
    return practice_condition_mapping

@st.cache_resource
def initialize_llm_models():
    """Cache LLM model initialization to avoid repeated API setup"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None
    
    classification_llm = ChatOpenAI(
        model="gpt-4-0613",
        temperature=0.1,
        openai_api_key=api_key,
        max_retries=2
    )
    generation_llm = ChatOpenAI(
        model="gpt-4-0613",
        temperature=0.7,
        openai_api_key=api_key,
        max_retries=2
    )
    return classification_llm, generation_llm

@st.cache_data
def get_common_words():
    """Precompute common word sets for validation"""
    return {
        '이', '그', '저', '것', '수', '있', '없', '는', '을', '를', '이', '가', '에', '의', '로', '으로', 
        '와', '과', '어떤', '어떻게', '왜', '무엇', '언제', '어디서', '어떠한', '그런', '이런', '저런',
        '하는', '되는', '있는', '없는', '같은', '다른', '새로운', '기존', '현재', '미래', '과거',
        '대한', '위한', '통해', '따라', '관련', '문제', '방법', '방식', '경우', '상황', '조건',
        '결과', '영향', '효과', '중요', '필요', '가능', '연구', '분석', '탐구', '제안', '개발',
        '창조', '혁신', '아이디어', '해결', '답', '질문', '생각', '고려', '검토', '평가'
    }

@lru_cache(maxsize=1000)
def get_content_words(text):
    """Cache content word extraction"""
    common_words = get_common_words()
    words = set(text.replace('?', '').replace('.', '').replace(',', '').lower().split())
    return words - common_words

def create_bloom_classification_chain(llm):
    """Create a chain for classifying questions according to Bloom's taxonomy with structured output."""
    
    # Create output parser
    parser = PydanticOutputParser(pydantic_object=BloomClassification)
    
    # Create example selector for few-shot prompting
    example_selector = LengthBasedExampleSelector(
        examples=BLOOM_CLASSIFICATION_EXAMPLES,
        example_prompt=PromptTemplate(
            input_variables=["paragraph", "question", "bloom_level"],
            template="Paragraph: {paragraph}\nQuestion: {question}\nBloom Level: {bloom_level}"
        ),
        max_length=2000,
    )
    
    # Create few-shot prompt template
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate(
            input_variables=["paragraph", "question", "bloom_level"],
            template="Paragraph: {paragraph}\nQuestion: {question}\nBloom Level: {bloom_level}"
        ),
        prefix="""다음은 Bloom's Taxonomy를 사용하여 질문을 분류하는 예시들입니다:

Bloom's Taxonomy 6단계:
1. 기억: 텍스트 내용을 기억하기 위한 질문 
2. 이해: 텍스트 내용을 바탕으로 대답할 수 있는 질문; 사실이나 이해, 정의에 대한 질문
3. 적용: 텍스트에 대해 추가적인 내용을 질문하거나 (방법, 선행문헌 등) 비슷하지만 다른 상황에 적용하는 질문
4. 분석: 텍스트의 내용 요소 간 연결 관계, 인과 관계 등을 묻는 질문, 텍스트 및 저자들의 의도를 묻는 질문
5. 평가: (배경지식을 활용하여) 텍스트에 대한 판단 및 비평을 제안하는 질문
6. 창조: 창의적인 연구 가설 또는 연구를 할 수 있는 새로운 방향을 제안하는 질문

예시들:""",
        suffix="""이제 다음 질문을 분류해주세요.

Paragraph: {paragraph}
Question: {question}

{format_instructions}

분류 결과:""",
        input_variables=["paragraph", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return LLMChain(
        llm=llm,
        prompt=few_shot_prompt,
        output_key="bloom_classification",
        output_parser=parser
    )

def create_related_question_generation_chain(llm):
    """Create a chain for generating related questions using structured output."""
    
    # Create output parser
    parser = PydanticOutputParser(pydantic_object=QuestionSuggestion)
    
    example_selector = LengthBasedExampleSelector(
        examples=RELATED_QUESTION_EXAMPLES,
        example_prompt=PromptTemplate(
            input_variables=["paragraph", "user_question", "suggested_question"],
            template="Paragraph: {paragraph}\nUser Question: {user_question}\nSuggested Question: {suggested_question}"
        ),
        max_length=1500,
    )
    
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate(
            input_variables=["paragraph", "user_question", "suggested_question"],
            template="Paragraph: {paragraph}\nUser Question: {user_question}\nSuggested Question: {suggested_question}"
        ),
        prefix="""다음은 사용자의 질문 요소를 활용해 '창조' 수준의 질문을 제안하는 예시들입니다:

중요 지침: 
- 반드시 사용자의 원래 질문에서 핵심 단어나 개념을 포함해야 합니다
- 사용자 질문을 확장하고 발전시키는 방향으로 작성하세요
- 완전히 새로운 주제로 바꾸지 마세요
- 사용자의 관심사와 접근법을 더 깊이 탐구하세요

예시들:""",
        suffix="""이제 다음 조건을 반드시 *모두* 따라 새로운 질문을 하나만 제안해주세요:

핵심 원칙: 학습자의 기존 질문을 발전시키고 확장하는 방향으로 질문을 구성하세요.

조건:
1. 학습자가 기존에 제시한 질문(question)의 핵심 키워드와 주제를 반드시 포함해야 함
2. 기존 질문에서 제기한 관점이나 접근법을 더 깊이 있게 탐구하는 방향
3. 기존 질문 + paragraph의 새로운 내용을 결합하여 확장된 질문 구성
4. Bloom's taxonomy에서 '창조' 수준의 질문 (새롭고 창의적인 연구 문제를 제안)
5. 대학교 학부생 수준에서 이해 가능해야 함
6. 질문은 한국어로 한 문장이어야 함 (글자수 65-75자)
7. 물음표로 끝나야 함

금지사항:
- 기존 질문과 완전히 다른 주제로 바꾸는 것
- 기존 질문의 핵심 개념을 무시하는 것
- 기존 질문보다 단순한 수준의 질문

중요: 학습자의 원래 질문 "{question}"의 핵심 요소를 반드시 포함하고 발전시켜야 합니다.

Paragraph: {paragraph}
User Question: {question}

{format_instructions}

새로운 질문 (기존 질문을 발전시킨 버전):""",
        input_variables=["paragraph", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return LLMChain(
        llm=llm,
        prompt=few_shot_prompt,
        output_key="question_suggestion",
        output_parser=parser
    )

def create_unrelated_question_generation_chain(llm):
    """Create a chain for generating unrelated questions using structured output."""
    
    # Create output parser
    parser = PydanticOutputParser(pydantic_object=QuestionSuggestion)
    
    example_selector = LengthBasedExampleSelector(
        examples=UNRELATED_QUESTION_EXAMPLES,
        example_prompt=PromptTemplate(
            input_variables=["paragraph", "user_question", "suggested_question"],
            template="Paragraph: {paragraph}\nUser Question: {user_question}\nSuggested Question: {suggested_question}"
        ),
        max_length=1500,
    )
    
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate(
            input_variables=["paragraph", "user_question", "suggested_question"],
            template="Paragraph: {paragraph}\nUser Question: {user_question}\nSuggested Question: {suggested_question}"
        ),
        prefix="""다음은 사용자의 질문과 무관하게 paragraph만을 기반으로 '창조' 수준의 질문을 제안하는 예시들입니다:

중요 지침:
- 사용자의 원래 질문에서 사용된 단어나 개념을 절대 사용하지 마세요
- 사용자 질문과는 완전히 다른 각도에서 접근하세요
- 사용자 질문을 발전시키거나 확장하지 마세요
- paragraph의 다른 측면이나 요소에 집중하세요

예시들:""",
        suffix="""이제 다음 조건을 반드시 *모두* 따라 새로운 질문을 하나만 제안해주세요:

핵심 원칙: 제시된 Paragraph 내에서, 학습자의 기존 질문과는 완전히 다른 관점을 탐구하는 질문을 구성하세요.

조건:
1. 반드시 제시된 Paragraph의 내용과 직접 관련된 질문이어야 함
2. 학습자가 기존에 제시한 질문(question)의 키워드, 주제, 접근법을 일절 사용하지 말 것
3. Paragraph에서 기존 질문이 다루지 않은 완전히 다른 측면이나 요소를 선택
4. 같은 텍스트 내의 다른 개념, 인물, 시대, 방법론, 분야 등에 집중
5. Bloom's taxonomy에서 '창조' 수준의 질문 (새롭고 창의적인 연구 문제를 제안)
6. 대학교 학부생 수준에서 이해 가능해야 함
7. 질문은 한국어로 한 문장이어야 함 (글자수 65-75자)
8. 물음표로 끝나야 함

금지사항:
- Paragraph 범위를 벗어나 완전히 다른 주제로 가는 것
- 기존 질문에서 언급된 개념이나 단어 재사용
- Paragraph에 없는 내용을 추가하는 것

전략: Paragraph를 다시 읽고, 사용자가 주목하지 않은 다른 요소(인물, 시대적 배경, 다른 개념, 응용 분야, 사회적 함의 등)를 찾아 질문하세요.

중요: 학습자의 원래 질문 "{question}"과는 완전히 무관하지만, Paragraph 내용에는 반드시 기반해야 합니다.

Paragraph: {paragraph}
User Question: {question}

{format_instructions}

새로운 질문 (기존 질문과 무관한 새로운 관점):""",
        input_variables=["paragraph", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return LLMChain(
        llm=llm,
        prompt=few_shot_prompt,
        output_key="question_suggestion",
        output_parser=parser
    )

def get_fallback_question(feedback_type, original_question):
    """Generate appropriate fallback questions when validation fails."""
    if feedback_type == "related":
        # Extract a key concept from original question for fallback
        words = original_question.replace('?', '').split()
        content_words = [w for w in words if len(w) > 2 and w not in ['어떤', '어떻게', '무엇', '왜']]
        if content_words:
            key_concept = content_words[0]
            return f"{key_concept}을 바탕으로 새로운 연구 방향을 제안해볼 수 있을까?"
        else:
            return "이 개념을 바탕으로 새로운 연구 방향을 제안해볼 수 있을까?"
    else:  # unrelated
        # Paragraph-grounded fallback questions that stay within the text scope
        fallback_questions = [
            "이 주제의 다른 측면을 새롭게 탐구할 수 있는 방법은 무엇일까?",
            "텍스트에서 다루지 않은 관련 요소를 발전시킬 수 있을까?",
            "이 개념을 다른 방향으로 확장해볼 수 있는 방안은 무엇일까?",
            "텍스트 내 다른 관점에서 새로운 접근법을 제안할 수 있을까?"
        ]
        import random
        return random.choice(fallback_questions)

def get_bloom_classification_with_fallback(llm, paragraph, question, max_retries=2):
    """Get Bloom classification with optimized retry logic"""
    classification_chain = create_bloom_classification_chain(llm)
    
    for attempt in range(max_retries):
        try:
            result = classification_chain.run({"paragraph": paragraph, "question": question})
            
            # Extract bloom level
            if hasattr(result, 'bloom_level'):
                return result.bloom_level
            elif isinstance(result, dict) and 'bloom_level' in result:
                return result['bloom_level']
            else:
                bloom_level = str(result).strip()
                if bloom_level:
                    return bloom_level
                    
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Classification failed after {max_retries} attempts: {e}")
            continue
    
    return "기억"  # Default fallback

def generate_question_without_validation(llm, paragraph, question, feedback_type, max_retries=3):
    """Generate question without validation but with metrics collection"""
    
    # Create appropriate chain
    if feedback_type == "related":
        chain = create_related_question_generation_chain(llm)
    else:
        chain = create_unrelated_question_generation_chain(llm)
    
    for attempt in range(max_retries):
        try:
            result = chain.run({"paragraph": paragraph, "question": question})
            
            # Extract question
            if hasattr(result, 'suggested_question'):
                suggested_question = result.suggested_question
            elif isinstance(result, dict) and 'suggested_question' in result:
                suggested_question = result['suggested_question']
            else:
                suggested_question = str(result).strip()
            
            # Basic format validation (just ensure it's not empty and has question mark)
            if suggested_question and len(suggested_question.strip()) > 0:
                if not suggested_question.endswith('?'):
                    suggested_question += '?'
                return suggested_question
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            continue
    
    # Fallback if all attempts failed
    return get_fallback_question(feedback_type, question)

def calculate_question_metrics(original_question, suggested_question, paragraph):
    """Calculate relatedness and other metrics for storage without validation"""
    
    # Calculate relatedness score
    original_content = get_content_words(original_question)
    suggested_content = get_content_words(suggested_question)
    
    if not original_content:
        relatedness_score = 0.0
    else:
        # Calculate overlap ratios
        overlap = len(original_content & suggested_content)
        overlap_ratio = overlap / len(original_content)
        
        # Concept overlap check
        concept_overlap = sum(1 for word in original_content if len(word) > 2 and word in suggested_question.lower())
        concept_ratio = concept_overlap / len(original_content)
        
        relatedness_score = max(overlap_ratio, concept_ratio)
    
    # Calculate paragraph relevance
    paragraph_content = get_content_words(paragraph)
    question_content = get_content_words(suggested_question)
    
    if not question_content:
        paragraph_relevance = 0.0
    else:
        overlap = len(paragraph_content & question_content)
        paragraph_relevance = overlap / len(question_content)
    
    # Calculate length
    question_length = len(suggested_question)
    
    # Calculate word count
    question_word_count = len(suggested_question.split())
    
    return {
        'relatedness_score': round(relatedness_score, 3),
        'paragraph_relevance': round(paragraph_relevance, 3),
        'question_length': question_length,
        'question_word_count': question_word_count,
        'ends_with_question_mark': suggested_question.endswith('?'),
        'is_empty': len(suggested_question.strip()) == 0
    }

def handle_api_error(error, feedback_type):
    """Centralized API error handling"""
    error_msg = str(error)
    if "insufficient_quota" in error_msg or "quota" in error_msg.lower():
        return "OpenAI API quota exceeded. Using mock response: '기억' 수준의 질문을 작성하셨군요.\n'이 내용을 바탕으로 새로운 아이디어를 제안해보세요?'와 같은 질문으로 수정하는 것은 어떨까요?"
    else:
        return f"Error generating AI feedback: {error_msg}"

# Function to get AI feedback using LangChain
def get_ai_feedback(question, paragraph_data, practice_mode=False, baseline_mode=False):
    """
    Optimized AI feedback generation without validation but with metrics collection
    paragraph_data should be a dict with 'index', 'content', 'genre' keys
    """
    
    if baseline_mode:
        # No feedback for baseline mode
        return "Baseline mode - no feedback"
    
    paragraph_index = paragraph_data['index']
    paragraph_content = paragraph_data['content']
    
    if practice_mode:
        # For practice mode, use balanced feedback assignment
        feedback_type = st.session_state.practice_condition_mapping.get(st.session_state.iteration, "related")
    else:
        feedback_type = st.session_state.condition_mapping.get(paragraph_index, "related")
    
    try:
        # Get cached LLM models
        classification_llm, generation_llm = initialize_llm_models()
        if not classification_llm:
            return "Error: OpenAI API key not found."
        
        # STEP 1: Classification (always needed)
        bloom_level = get_bloom_classification_with_fallback(classification_llm, paragraph_content, question)
        
        # STEP 2: Generate suggestion
        suggested_question = generate_question_without_validation(
            generation_llm, paragraph_content, question, feedback_type
        )
        
        # Calculate metrics for storage (but don't use for validation)
        question_metrics = calculate_question_metrics(question, suggested_question, paragraph_content)
        
        final_response = f"'{bloom_level}' 수준의 질문을 작성하셨군요.\n'{suggested_question}'와 같은 질문으로 수정하는 것은 어떨까요?"
        
        # Store metrics in session state for later CSV inclusion
        if question_metrics:
            st.session_state.current_iteration_data.update({
                'suggested_question_metrics': question_metrics,
                'feedback_type': feedback_type
            })
        
        # Log execution details
        log_event("AI feedback generated", {
            "bloom_level": bloom_level,
            "suggested_question": suggested_question,
            "feedback_type": feedback_type,
            "paragraph_index": paragraph_index,
            "paragraph_genre": paragraph_data.get('genre', 'unknown'),
            "question_metrics": question_metrics,
            "practice_mode": practice_mode,
            "baseline_mode": baseline_mode
        })
        
        return final_response
        
    except Exception as e:
        return handle_api_error(e, feedback_type)

def get_session_value(key, default=None):
    """Helper to safely get session state values"""
    return st.session_state.get(key, default)

def set_session_value(key, value):
    """Helper to set session state values"""
    st.session_state[key] = value

def create_widget_key(base_name, iteration=None):
    """Create consistent widget keys"""
    if iteration is None:
        iteration = get_session_value('iteration', 0)
    return f"{base_name}_{iteration}"

class EventLogger:
    def __init__(self):
        self.pending_events = []
        self.batch_size = 5  # Smaller batch for experiment context
    
    def add_event(self, event_description, data=None):
        """Add event to batch"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        
        log_entry = {
            "timestamp": timestamp,
            "iteration": get_session_value('iteration', 0),
            "stage": get_session_value('stage', 'unknown'),
            "event": event_description
        }
        
        if data:
            log_entry["data"] = data
        
        self.pending_events.append(log_entry)
        
        # Flush if batch is full or for critical events
        if (len(self.pending_events) >= self.batch_size or 
            "completed" in event_description.lower() or
            "error" in event_description.lower()):
            self.flush()
    
    def flush(self):
        """Write all pending events to session state"""
        if 'event_log' not in st.session_state:
            st.session_state.event_log = []
        
        st.session_state.event_log.extend(self.pending_events)
        self.pending_events.clear()

# Initialize global logger
if 'logger' not in st.session_state:
    st.session_state.logger = EventLogger()
        
# Initialize session state variables if they don't exist
def initialize_session_state():
    if 'started' not in st.session_state:
        st.session_state.started = False
    
    if 'iteration' not in st.session_state:
        st.session_state.iteration = 0
    
    if 'stage' not in st.session_state:
        st.session_state.stage = "start"
    
    # Initialize timing variables
    if 'stage_timers' not in st.session_state:
        st.session_state.stage_timers = {}
    
    if 'responses' not in st.session_state:
        st.session_state.responses = []
    
    if 'practice_responses' not in st.session_state:
        st.session_state.practice_responses = []
    
    # Initialize current iteration data storage
    if 'current_iteration_data' not in st.session_state:
        st.session_state.current_iteration_data = {}
    
    # Initialize baseline mode flag
    if 'baseline_mode' not in st.session_state:
        st.session_state.baseline_mode = False
    
    if 'baseline_completed' not in st.session_state:
        st.session_state.baseline_completed = False
    
    # Initialize practice mode flag
    if 'practice_mode' not in st.session_state:
        st.session_state.practice_mode = False
    
    if 'practice_completed' not in st.session_state:
        st.session_state.practice_completed = False

# Function to start stage timer
def start_stage_timer(stage_name):
    st.session_state.stage_timers[f"{stage_name}_start"] = time.time()

# Function to end stage timer and calculate duration
def end_stage_timer(stage_name):
    start_key = f"{stage_name}_start"
    duration_key = f"{stage_name}_duration"
    
    if start_key in st.session_state.stage_timers:
        duration = time.time() - st.session_state.stage_timers[start_key]
        st.session_state.stage_timers[duration_key] = duration
        return duration
    return 0

# Function to advance to the next stage
def next_stage(next_stage_name):
    # End timer for current stage
    end_stage_timer(st.session_state.stage)
    
    # Log the stage transition
    log_event(f"Stage transition: {st.session_state.stage} -> {next_stage_name}")
    st.session_state.stage = next_stage_name

    time.sleep(0.1)
    
    # Start timer for next stage
    start_stage_timer(next_stage_name)
    
    st.rerun()

# Function to handle the start of a new iteration
def start_iteration():
    if st.session_state.baseline_mode:
        # In baseline mode - only one iteration
        st.session_state.stage = "baseline_screen"
        start_stage_timer("baseline_screen")
        send_marker("baseline_start")
        log_event("Baseline session started")
    elif st.session_state.practice_mode:
        # In practice mode
        if st.session_state.iteration >= 2:
            st.session_state.stage = "practice_completed"
            log_event("Practice session completed")
        else:
            st.session_state.stage = "show_paragraph"
            start_stage_timer("show_paragraph")
            send_marker("paragraph_start")
            log_event("Practice iteration started", {"iteration_number": st.session_state.iteration})
    else:
        # In main experiment - check if experiment_paragraphs exists
        if hasattr(st.session_state, 'experiment_paragraphs'):
            if st.session_state.iteration >= len(st.session_state.experiment_paragraphs):
                st.session_state.stage = "completed"
                log_event("Experiment completed")
            else:
                st.session_state.stage = "show_paragraph"
                start_stage_timer("show_paragraph")
                send_marker("paragraph_start")
                log_event("Iteration started", {"iteration_number": st.session_state.iteration})
        else:
            # If experiment_paragraphs doesn't exist yet, stay in current stage
            # This happens during baseline completion before main experiment starts
            log_event("Experiment paragraphs not yet initialized")
            pass

# Function to handle baseline completion
def baseline_completed():
    send_marker("baseline_end")
    end_stage_timer("baseline_screen")
    
    # Log baseline completion
    log_event("Baseline session completed", {
        "baseline_duration": st.session_state.stage_timers.get("baseline_screen_duration", 0)
    })
    
    # Switch to baseline completion stage (NEW!)
    st.session_state.baseline_mode = False
    st.session_state.baseline_completed = True
    st.session_state.stage = "baseline_complete"  # <-- CHANGED: Go to completion stage first
    st.session_state.stage_timers = {}
    st.session_state.current_iteration_data = {}
    
    log_event("Moving to baseline completion stage")  # <-- CHANGED: Log the new stage
    
    # Create practice condition assignment
    st.session_state.practice_condition_mapping = create_practice_condition_assignment(
        st.session_state.participant_id
    )

    log_event("Practice session prepared", {
        "practice_condition_mapping": st.session_state.practice_condition_mapping
    })

# Function to handle the completion of viewing paragraph
def paragraph_viewed():
    send_marker("paragraph_end")
    send_marker("question_input_start")
    next_stage("ask_question")

# Function to log textarea focus events
def log_textarea_focus(textarea_type):
    """Log when a textarea is focused/clicked"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    if textarea_type == "edit_question":
        # Store the focus time for edit question specifically
        st.session_state.edit_textarea_focus_time = time.time()
        send_marker("edit_textarea_focus")
    elif textarea_type == "question_input":
        # Store the focus time for question input specifically
        st.session_state.question_input_focus_time = time.time()
    
    log_event(f"Textarea focus: {textarea_type}", {
        "focus_timestamp": timestamp,
        "focus_time": time.time()
    })

# Function to handle question submission
def submit_question():
    # Get the question from session state (it should exist now)
    question = st.session_state.get('user_question', '')
    question_comments = st.session_state.get('question_comments', '')
    
    # Enhanced validation
    if not question.strip():
        st.error("질문을 입력해주세요.")
        return
    
    # Check if the question is only a question mark (with optional whitespace)
    if question.strip() == '?':
        st.error("질문을 입력해주세요.")
        return
    
    # Calculate question input interaction time (from first focus to submission)
    question_input_interaction_time = None
    if hasattr(st.session_state, 'question_input_focus_time'):
        question_input_interaction_time = time.time() - st.session_state.question_input_focus_time
    
    # Store the question in current iteration data for persistence
    st.session_state.current_iteration_data['user_question'] = question
    st.session_state.current_iteration_data['question_comments'] = question_comments
    st.session_state.current_iteration_data['question_input_interaction_time'] = question_input_interaction_time
    
    send_marker("question_input_end")
    
    # Get paragraph information
    if st.session_state.practice_mode:
        current_paragraph_data = st.session_state.practice_paragraphs[st.session_state.iteration]
    else:
        current_paragraph_data = st.session_state.experiment_paragraphs[st.session_state.iteration]
    
    # Log the submitted question with paragraph information
    log_event_batched("Question submitted", {
        "question": question,
        "question_comments": question_comments,
        "question_input_interaction_time": question_input_interaction_time,
        "iteration": st.session_state.iteration,
        "paragraph_index": current_paragraph_data['index'],
        "paragraph_genre": current_paragraph_data['genre'],
        "paragraph": current_paragraph_data['content'],
        "practice_mode": st.session_state.practice_mode,
        "baseline_mode": st.session_state.baseline_mode
    })
    
    # Get AI feedback
    send_marker("feedback_start")
    feedback = get_ai_feedback(
        question, 
        current_paragraph_data,
        practice_mode=st.session_state.practice_mode,
        baseline_mode=st.session_state.baseline_mode
    )
    send_marker("feedback_end")
    
    # Store the feedback
    st.session_state.current_iteration_data['feedback'] = feedback
    
    # Log the feedback
    feedback_type = st.session_state.current_iteration_data.get('feedback_type', 'unknown')
    log_event("AI feedback generated", {
        "feedback": feedback,
        "feedback_type": feedback_type,
        "paragraph_index": current_paragraph_data['index'],
        "paragraph_genre": current_paragraph_data['genre'],
        "practice_mode": st.session_state.practice_mode,
        "baseline_mode": st.session_state.baseline_mode
    })
    
    next_stage("show_feedback")

# Function to handle survey submission
def submit_survey():
    send_marker("survey_end")
    
    # Get survey responses
    curiosity = st.session_state.get('curiosity')
    relatedness = st.session_state.get('relatedness')
    accept_feedback = st.session_state.get('accept_feedback')
    feedback_comments = st.session_state.get('feedback_comments', '')
    survey_comments = st.session_state.get('survey_comments', '')
    
    # Validate that required fields are filled
    if curiosity is None:
        st.error("Please rate your curiosity level before proceeding.")
        return
    if accept_feedback is None:
        st.error("Please indicate whether you accept the feedback before proceeding.")
        return
    if relatedness is None:
        st.error("Please rate the relatedness before proceeding.")
        return
    
    # Store in current iteration data
    st.session_state.current_iteration_data['curiosity'] = curiosity
    st.session_state.current_iteration_data['relatedness'] = relatedness
    st.session_state.current_iteration_data['accept_feedback'] = accept_feedback
    st.session_state.current_iteration_data['feedback_comments'] = feedback_comments
    st.session_state.current_iteration_data['survey_comments'] = survey_comments
    
    # Get paragraph information
    if st.session_state.practice_mode:
        current_paragraph_data = st.session_state.practice_paragraphs[st.session_state.iteration]
    else:
        current_paragraph_data = st.session_state.experiment_paragraphs[st.session_state.iteration]
    
    # Log the survey responses
    survey_data = {
        "curiosity": curiosity,
        "relatedness": relatedness,
        "accept_feedback": accept_feedback,
        "feedback_comments": feedback_comments,
        "survey_comments": survey_comments,
        "iteration": st.session_state.iteration,
        "paragraph_index": current_paragraph_data['index'],
        "paragraph_genre": current_paragraph_data['genre'],
        "feedback_type": st.session_state.current_iteration_data.get('feedback_type', 'unknown'),
        "practice_mode": st.session_state.practice_mode,
        "baseline_mode": st.session_state.baseline_mode
    }
    log_event_batched("Survey submitted", survey_data)
    
    # Go to edit question stage
    send_marker("edit_start")
    next_stage("edit_question")

# Function to handle edited question submission
def submit_edited_question():
    send_marker("edit_end")
    
    # Get the edited question and comments
    edited_question = st.session_state.get('edited_question', '')
    edit_comments = st.session_state.get('edit_comments', '')
    
    if not edited_question.strip():
        st.error("Please enter a question before proceeding.")
        return
    
    # Store in current iteration data
    st.session_state.current_iteration_data['edited_question'] = edited_question
    st.session_state.current_iteration_data['edit_comments'] = edit_comments
    
    # Log the edited question
    log_event("Edited question submitted", {
        "edited_question": edited_question,
        "edit_comments": edit_comments,
        "practice_mode": st.session_state.practice_mode,
        "baseline_mode": st.session_state.baseline_mode
    })
    
    # Get the current paragraph data
    if st.session_state.practice_mode:
        current_paragraph_data = st.session_state.practice_paragraphs[st.session_state.iteration]
    else:
        current_paragraph_data = st.session_state.experiment_paragraphs[st.session_state.iteration]
    
    # End the current stage timer BEFORE calculating durations
    end_stage_timer(st.session_state.stage)
    
    # Calculate stage durations
    stage_durations = {}
    for stage in ['show_paragraph', 'ask_question', 'show_feedback', 'survey', 'edit_question']:
        duration_key = f"{stage}_duration"
        if duration_key in st.session_state.stage_timers:
            stage_durations[f"{stage}_time_seconds"] = st.session_state.stage_timers[duration_key]
    
    # Calculate edit textarea interaction time
    edit_textarea_interaction_time = None
    if hasattr(st.session_state, 'edit_textarea_focus_time'):
        edit_textarea_interaction_time = time.time() - st.session_state.edit_textarea_focus_time
    
    # Get metrics if they exist
    metrics = st.session_state.current_iteration_data.get('suggested_question_metrics', {})
    
    # Store all the data for this iteration
    iteration_data = {
        "iteration": st.session_state.iteration,
        "paragraph": current_paragraph_data['content'],
        "paragraph_index": current_paragraph_data['index'],
        "paragraph_genre": current_paragraph_data['genre'],
        "feedback_type": st.session_state.current_iteration_data.get('feedback_type', 'unknown'),
        "original_question": st.session_state.current_iteration_data.get('user_question', ''),
        "question_comments": st.session_state.current_iteration_data.get('question_comments', ''),
        "question_input_interaction_time_seconds": st.session_state.current_iteration_data.get('question_input_interaction_time'),
        "feedback": st.session_state.current_iteration_data.get('feedback', ''),
        "feedback_comments": st.session_state.current_iteration_data.get('feedback_comments', ''),
        "curiosity": st.session_state.current_iteration_data.get('curiosity'),
        "relatedness": st.session_state.current_iteration_data.get('relatedness'),
        "accept_feedback": st.session_state.current_iteration_data.get('accept_feedback'),
        "survey_comments": st.session_state.current_iteration_data.get('survey_comments', ''),
        "edited_question": edited_question,
        "edit_comments": edit_comments,
        "edit_textarea_interaction_time_seconds": edit_textarea_interaction_time,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # Add question metrics to CSV
        "suggested_question_relatedness_score": metrics.get('relatedness_score'),
        "suggested_question_paragraph_relevance": metrics.get('paragraph_relevance'),
        "suggested_question_length": metrics.get('question_length'),
        "suggested_question_word_count": metrics.get('question_word_count'),
        "suggested_question_ends_with_question_mark": metrics.get('ends_with_question_mark'),
        "suggested_question_is_empty": metrics.get('is_empty'),
        **stage_durations  # Add all stage durations
    }
    
    if st.session_state.practice_mode:
        st.session_state.practice_responses.append(iteration_data)
    else:
        st.session_state.responses.append(iteration_data)
    
    # Clear current iteration data and stage timers for next iteration
    st.session_state.current_iteration_data = {}
    st.session_state.stage_timers = {}

    # Clear edit textarea focus time
    if hasattr(st.session_state, 'edit_textarea_focus_time'):
        delattr(st.session_state, 'edit_textarea_focus_time')
    
    # Clear question input focus time
    if hasattr(st.session_state, 'question_input_focus_time'):
        delattr(st.session_state, 'question_input_focus_time')
    
    # Reset widget keys by removing them from session state
    widget_keys_to_reset = [
        'user_question', 'edited_question',
        'paragraph_comments', 'question_comments', 'feedback_comments', 
        'survey_comments', 'edit_comments', 'curiosity', 'relatedness', 'accept_feedback'
    ]
    
    for key in widget_keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    # Move to next iteration
    if hasattr(st.session_state, 'logger'):
        st.session_state.logger.flush()
    st.session_state.iteration += 1
    start_iteration()

# Main app
def main():
    st.markdown("## [파일럿] 생성형 AI의 피드백 유형이 질문 수정에 미치는 영향")
    
    # Initialize session state
    initialize_session_state()
    
    # Handle baseline screen FIRST, before anything else
    if st.session_state.get('stage') == "baseline_screen":
        # Send the start marker
        send_marker("baseline_start")
        log_event("Baseline session started")
        
        # Create an empty container for the baseline display
        baseline_container = st.empty()
        
        # Show the baseline screen for 30 seconds
        duration = 30  # 30 seconds
        
        # Display the '+' symbol
        with baseline_container.container():
            st.markdown("""
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                height: 150vh;
                font-size: 72px;
                font-weight: bold;
            ">
            +
            </div>
            """, unsafe_allow_html=True)
        
        # Sleep for the full duration
        time.sleep(duration)
        
        # Clear the container
        baseline_container.empty()
        
        # Automatically proceed to next stage
        baseline_completed()
        st.rerun()
        return  # Don't render anything else
    
    # Check if the experiment has started
    if not st.session_state.started:
        st.write("Welcome to the experiment!")
        
        # Experiment instructions and Bloom's taxonomy explanation
        st.markdown("""
        ### 실험 안내
        
        본 실험은 학습자가 질문을 만든 후 생성형 AI로부터 질문에 대한 피드백을 제공받았을 때, 
        피드백을 수용하고 질문을 수정하는 인지적 과정을 탐구합니다.
        
        ---
        
        실험은 사전설문(10분), EEG 장비 착용(30분), 베이스라인 측정(30초), 2개의 연습 문제(5분), 그리고 본 실험(55분)으로 구성되어 있습니다.
        사전 설문을 시작하려면 아래에 참여자 ID를 입력하고, '실험 시작' 버튼을 눌러주세요.
        """)
        
        participant_id = st.text_input("참여자 ID를 입력해주세요(예: pilot1):")
        
        if st.button("실험 시작") and participant_id:
            st.session_state.participant_id = participant_id
            st.session_state.started = True
            st.session_state.stage = "pretest_survey"
            
            log_event("Experiment started", {
                "participant_id": participant_id,
                "starting_with_pretest": True
            })
            
            st.rerun()
    
    else:
        # Handle pretest survey
        if st.session_state.stage == "pretest_survey":
            st.header("사전 설문조사")
            st.write("실험을 시작하기 전에 몇 가지 질문에 답해주세요.")
            
            with st.form("pretest_survey_form"):
                # I. 인적사항
                st.subheader("I. 인적사항")
                                
                # 1. 성별
                gender = st.radio("귀하의 성별을 선택하십시오.", ["남", "여"], horizontal=True, index=None)
                
                # 2. 나이
                age = st.number_input("귀하의 현재 만 나이를 기입하십시오.", min_value=18, max_value=100, value=None)
                
                # 3. 전공 및 학력 - Using st.data_editor
                st.text("귀하의 전공 및 학력 사항을 모두 기입하십시오.")
                st.text("필요한 경우 행을 추가하여 여러 전공/학력을 입력할 수 있습니다.")
                st.caption("칸을 더블클릭하여 정보를 입력해주세요.")
                
                # Initialize default education data
                if 'education_data' not in st.session_state:
                    st.session_state.education_data = pd.DataFrame({
                        '전공명': [''],
                        '학위명': [''],
                        '졸업여부': ['']
                    })
                
                # Use data editor for education information
                education_df = st.data_editor(
                    st.session_state.education_data,
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "전공명": st.column_config.TextColumn(
                            "전공명",
                            help="전공명을 입력해주세요 (예: 교육학과)",
                            max_chars=50,
                            required=True
                        ),
                        "학위명": st.column_config.SelectboxColumn(
                            "학위명",
                            help="학위명을 선택해주세요",
                            options=["학사", "석사", "박사"],
                            required=True
                        ),
                        "졸업여부": st.column_config.SelectboxColumn(
                            "졸업여부",
                            help="졸업 여부를 선택해주세요",
                            options=["재학", "졸업", "휴학", "수료"],
                            required=True
                        )
                    },
                    key="education_editor"
                )
                
                # Update session state with edited data
                st.session_state.education_data = education_df
                
                # II. 인공지능 활용 경험
                st.subheader("II. 인공지능 활용 경험")
                
                ai_frequency = st.number_input("1. 일주일 평균 인공지능 활용 횟수는 몇 회인가요?", min_value=0, value=0)
                ai_tools = st.text_input("2. 활용하시는 인공지능을 모두 적어주세요. (예: ChatGPT)")
                ai_usage = st.text_area("3. 주로 어떤 용도로 인공지능을 활용하시나요?", height=100)
                
                # III. 읽기 자기 효능감
                st.subheader("III. 읽기 자기 효능감")
                st.write("다음은 읽기 자기 효능감을 측정하는 문항입니다. 다음에 제시된 각 문항을 읽고 1(전혀 그렇지 않다)에서 5(매우 그렇다) 중 자신이 생각하는 바와 가장 일치하는 곳에 표시하세요.")
                
                reading_efficacy_items = [
                    "나는 학술적 논문이나 서적을 읽을 때 핵심 요지를 잘 파악한다.",
                    "나는 충분한 노력을 기울이면 학술적 논문이나 서적을 잘 이해할 수 있다.",
                    "학술적 논문이나 서적을 읽을 때, 나는 추가로 찾아볼 만한 관련된 자료를 잘 찾을 수 있다.",
                    "나는 학술적 논문이나 서적을 읽은 후 텍스트에 대한 질문에 잘 대답할 수 있다.",
                    "나는 읽은 각 문장을 잘 이해한다.",
                    "나는 학술적 논문이나 서적을 다 읽은 후에 중요한 내용을 잘 기억할 수 있다.",
                    "나는 학술적 논문이나 서적의 내용을 완전히 이해한 후에 비판적으로 검토할 수 있다.",
                    "나는 학술적 논문이나 서적을 읽으면서 모국어로 스스로의 언어를 사용해 재진술하여 필기를 할 수 있다.",
                    "나는 학술적 논문이나 서적을 이해하지 못하더라도, 강의를 들으면 해당 내용을 이해할 수 있다.",
                    "나는 하이라이트나 밑줄 치는 등의 방법을 활용해서 학술적 논문이나 서적에 대한 이해를 높일 수 있다.",
                    "나는 여러 학술적 논문이나 서적으로부터 가장 목적에 맞는 텍스트를 선택할 수 있다."
                ]
                
                reading_efficacy_responses = []
                for i, item in enumerate(reading_efficacy_items):
                    response = st.radio(
                        f"{i+1}. {item}",
                        ["1", "2", "3", "4", "5"],
                        index=None,
                        key=f"reading_efficacy_{i}",
                        horizontal=True,
                        help = "1 (전혀 그렇지 않다), 2 (그렇지 않다), 3 (보통이다), 4 (그렇다), 5 (매우 그렇다)"
                    )
                    reading_efficacy_responses.append(response)
                
                # IV. 호기심
                st.subheader("IV. 호기심")
                st.write("다음은 호기심에 대한 문항입니다. 다음에 제시된 각 문항을 읽고 1(전혀 그렇지 않다)에서 5(매우 그렇다) 중 자신이 생각하는 바와 가장 일치하는 곳에 표시하세요.")
                
                curiosity_items = [
                    "나는 새로운 상황에서 가능한 한 많은 정보를 적극적으로 찾아나선다.",
                    "나는 일상 속에서 발생하는 예측 불가능성을 즐긴다.",
                    "나는 복잡하거나 도전적인 과제를 할 때 가장 큰 성취감을 느낀다.",
                    "나는 어디를 가든 새로운 경험과 물건 등을 찾는다.",
                    "나는 도전적인 상황을 성장과 학습의 기회로 여긴다.",
                    "나는 조금은 두려운 것들을 경험하는 것을 좋아한다.",
                    "나는 항상 나와 세상에 대한 사고의 전환을 시켜줄 수 있는 경험을 찾는다.",
                    "나는 흥미진진하고 예측 불가능한 직업을 선호한다.",
                    "나는 스스로 도전하고 성장할 수 있는 기회를 자주 찾아나선다.",
                    "나는 낯선 사람, 사건, 장소를 포용하는 사람이다."
                ]
                
                curiosity_responses = []
                for i, item in enumerate(curiosity_items):
                    response = st.radio(
                        f"{i+1}. {item}",
                        ["1", "2", "3", "4", "5"],
                        index=None,
                        key=f"curiosity_{i}",
                        horizontal=True,
                        help = "1 (전혀 그렇지 않다), 2 (그렇지 않다), 3 (보통이다), 4 (그렇다), 5 (매우 그렇다)"
                    )
                    curiosity_responses.append(response)
                
                # V. 인공지능 태도
                st.subheader("V. 인공지능 태도")
                st.write("다음은 인공지능에 대한 태도를 측정하는 문항입니다. 다음에 제시된 각 문항을 읽고 1(전혀 그렇지 않다)에서 5(매우 그렇다) 중 자신이 평소 생각하는 바와 가장 일치하는 곳에 표시하세요.")
                
                ai_attitude_items = [
                    "나는 일상생활 속에서 인공지능을 사용하는 데 흥미를 가지고 있다.",
                    "나는 인공지능 사용이 재미있다.",
                    "인공지능은 많은 곳에서 유익하게 활용된다.",
                    "나는 인공지능을 내가 하는 일에 사용하고 싶다.",
                    "나는 인공지능을 비윤리적으로 사용하는 기관들이 있다고 생각한다.",
                    "나는 인공지능이 해롭다고 생각한다.",
                    "나는 인공지능이 위험하다고 생각한다.",
                    "인공지능은 사람을 감시하는 데 사용된다."
                ]
                
                ai_attitude_responses = []
                for i, item in enumerate(ai_attitude_items):
                    response = st.radio(
                        f"{i+1}. {item}",
                        ["1", "2", "3", "4", "5"],
                        index=None,
                        key=f"ai_attitude_{i}",
                        horizontal=True,
                        help = "1 (전혀 그렇지 않다), 2 (그렇지 않다), 3 (보통이다), 4 (그렇다), 5 (매우 그렇다)"
                    )
                    ai_attitude_responses.append(response)
                
                # VI. 인공지능 신뢰
                st.subheader("VI. 인공지능 신뢰")
                st.write("다음은 인공지능에 대한 신뢰를 측정하는 문항입니다. 다음에 제시된 각 문항을 읽고 1(전혀 그렇지 않다)에서 5(매우 그렇다) 중 자신이 평소 생각하는 바와 가장 일치하는 곳에 표시하세요.")
                
                ai_trust_items = [
                    "나는 인공지능 사용 시 부정적인 결과가 발생할 수 있다고 생각한다.",
                    "인공지능을 사용할 때 조심해야 한다.",
                    "인공지능과 상호작용하는 것은 위험하다.",
                    "나는 인공지능이 나를 위한 최선의 이익을 위해 행동할 것이라고 믿는다.",
                    "내가 도움이 필요하면 인공지능은 나를 돕기 위해 최선을 다할 것이다.",
                    "나는 인공지능이 나의 요구와 선호를 이해하는 데 관심이 있다고 믿는다.",
                    "인공지능을 사용하면 완전히 의존할 수 있을 것 같다.",
                    "나는 언제든지 인공지능에 의지할 수 있다.",
                    "나는 인공지능이 제시한 정보를 신뢰할 수 있다."
                ]
                
                ai_trust_responses = []
                for i, item in enumerate(ai_trust_items):
                    response = st.radio(
                        f"{i+1}. {item}",
                        ["1", "2", "3", "4", "5"],
                        index=None,
                        key=f"ai_trust_{i}",
                        horizontal=True,
                        help = "1 (전혀 그렇지 않다), 2 (그렇지 않다), 3 (보통이다), 4 (그렇다), 5 (매우 그렇다)"
                    )
                    ai_trust_responses.append(response)
                
                # Submit button
                submitted = st.form_submit_button("설문 제출")
                
                if submitted:
                    # Validate required fields
                    missing_fields = []
                    if gender is None:
                        missing_fields.append("성별")
                    if age is None:
                        missing_fields.append("나이")
                    
                    # Validate education data
                    education_valid = True
                    education_errors = []
                    
                    # Check if at least one row has complete information
                    complete_rows = 0
                    for idx, row in education_df.iterrows():
                        if pd.isna(row['전공명']) or row['전공명'].strip() == '':
                            if not (pd.isna(row['학위명']) and pd.isna(row['졸업여부'])):
                                education_errors.append(f"행 {idx+1}: 전공명이 비어있습니다")
                        elif pd.isna(row['학위명']) or row['학위명'] == '':
                            education_errors.append(f"행 {idx+1}: 학위명을 선택해주세요")
                        elif pd.isna(row['졸업여부']) or row['졸업여부'] == '':
                            education_errors.append(f"행 {idx+1}: 졸업여부를 선택해주세요")
                        else:
                            complete_rows += 1
                    
                    if complete_rows == 0:
                        education_errors.append("최소 하나의 전공/학력 정보를 완전히 입력해주세요")
                    
                    if education_errors:
                        missing_fields.append("전공 및 학력 정보")
                        education_valid = False
                    
                    # Check if all scale responses are completed
                    if None in reading_efficacy_responses:
                        missing_fields.append("읽기 자기 효능감 문항")
                    if None in curiosity_responses:
                        missing_fields.append("호기심 문항")
                    if None in ai_attitude_responses:
                        missing_fields.append("인공지능 태도 문항")
                    if None in ai_trust_responses:
                        missing_fields.append("인공지능 신뢰 문항")
                    
                    if missing_fields:
                        error_message = f"다음 항목들을 모두 작성해주세요: {', '.join(missing_fields)}"
                        if not education_valid:
                            error_message += f"\n전공 및 학력 오류:\n" + "\n".join(education_errors)
                        st.error(error_message)
                    else:
                        # Process and save pretest data
                        pretest_data = {
                            "participant_id": st.session_state.participant_id,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "gender": gender,
                            "age": age,
                            "ai_frequency_per_week": ai_frequency,
                            "ai_tools_used": ai_tools,
                            "ai_usage_purposes": ai_usage,
                        }
                        
                        # Add education data (serialize the complete education information)
                        education_records = []
                        for idx, row in education_df.iterrows():
                            if not (pd.isna(row['전공명']) or row['전공명'].strip() == ''):
                                education_records.append({
                                    'major': row['전공명'],
                                    'degree': row['학위명'],
                                    'graduation_status': row['졸업여부']
                                })
                        
                        pretest_data['education_records'] = str(education_records)  # Store as string for CSV compatibility
                        pretest_data['education_count'] = len(education_records)
                        
                        # For backward compatibility, also store the first record in separate fields
                        if education_records:
                            pretest_data['major'] = education_records[0]['major']
                            pretest_data['degree'] = education_records[0]['degree']
                            pretest_data['graduation_status'] = education_records[0]['graduation_status']
                        
                        # Add reading efficacy responses
                        for i, response in enumerate(reading_efficacy_responses):
                            pretest_data[f"reading_efficacy_{i+1}"] = response.split()[0] if response else None
                        
                        # Add curiosity responses
                        for i, response in enumerate(curiosity_responses):
                            pretest_data[f"curiosity_{i+1}"] = response.split()[0] if response else None
                        
                        # Add AI attitude responses
                        for i, response in enumerate(ai_attitude_responses):
                            pretest_data[f"ai_attitude_{i+1}"] = response.split()[0] if response else None
                        
                        # Add AI trust responses
                        for i, response in enumerate(ai_trust_responses):
                            pretest_data[f"ai_trust_{i+1}"] = response.split()[0] if response else None
                        
                        # Store in session state
                        st.session_state.pretest_data = pretest_data
                        
                        # Log the completion
                        log_event("Pretest survey completed", pretest_data)
                        
                        # Move to pretest completion stage
                        st.session_state.stage = "pretest_completed"
                        st.rerun()
        
        # Handle pretest completion
        elif st.session_state.stage == "pretest_completed":
            st.success("사전 설문조사가 완료되었습니다!")
            st.write("버튼을 눌러 설문 결과를 다운로드해주세요.")
            
            # Create download button for pretest data
            if hasattr(st.session_state, 'pretest_data'):
                pretest_df = pd.DataFrame([st.session_state.pretest_data])
                pretest_csv = pretest_df.to_csv(index=False)
                
                st.download_button(
                    label="사전 설문 결과 다운로드 (CSV)",
                    data=pretest_csv,
                    file_name=f"pretest_survey_{st.session_state.participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="pretest_download"
                )
            
            st.write("이제 베이스라인 측정을 시작하겠습니다.")
            
            if st.button("베이스라인 측정으로 이동"):
                st.session_state.baseline_mode = True
                st.session_state.iteration = 0
                st.session_state.stage = "baseline_ready"
                st.rerun()
        
        # Handle baseline ready stage
        elif st.session_state.stage == "baseline_ready":
            st.subheader("베이스라인 측정 준비")
            st.write("베이스라인 측정을 시작하려면 아래 버튼을 클릭해주세요.")
            st.write("버튼을 클릭한 후, 30초 동안 화면 중앙의 '+' 기호를 봐주세요.")
            
            if st.button("베이스라인 시작"):
                st.session_state.stage = "baseline_screen"
                st.session_state.baseline_start_time = time.time()
                send_marker("baseline_start")
                log_event("Baseline session started")
                st.rerun()
        
        # NOTE: baseline_screen is handled at the top of the main() function
        
        # Handle baseline completion screen
        elif st.session_state.stage == "baseline_complete":
            st.success("베이스라인 측정이 완료되었습니다!")
            st.write("30초 동안의 베이스라인 측정이 성공적으로 완료되었습니다.")
            
            if st.button("다음 단계로 이동", use_container_width=True):
                # Move to Bloom explanation stage
                st.session_state.stage = "bloom_explanation"
                log_event("Moving to Bloom explanation stage")
                st.rerun()
        
        # Handle Bloom's taxonomy explanation before practice
        elif st.session_state.stage == "bloom_explanation":
            st.header("실험 안내")
            st.write("연습 세션을 시작하기 전에 실험에 대한 안내를 읽어주세요.")
            
            st.markdown("""
            ### 실험 개요
            
            본 실험은 주어진 텍스트를 읽고 질문을 작성한 후, AI로부터 질문에 대한 피드백을 받는 과정으로 구성되어 있습니다. 
            AI는 여러분이 작성한 질문이 Bloom의 분류체계(Bloom's Taxonomy) 중 어느 수준에 해당하는지에 대한 피드백을 제공할 것입니다.
            
            ---
                    
            #### Bloom의 분류체계 6단계:
            
            **1. 기억**: 텍스트 내용을 기억하기 위한 질문\n
            **2. 이해**: 텍스트 내용을 바탕으로 대답할 수 있는 질문; 사실이나 이해, 정의에 대한 질문\n
            **3. 적용**: 텍스트에 대해 추가 내용을 질문하거나(방법, 선행문헌 등) 비슷하지만 다른 상황에 적용하는 질문\n
            **4. 분석**: 텍스트의 내용 요소 간 연결 관계, 인과 관계 등을 묻는 질문, 텍스트 및 저자들의 의도를 묻는 질문\n
            **5. 평가**: (배경지식을 활용하여) 텍스트에 대한 판단 및 비평을 제안하는 질문\n
            **6. 창조**: 창의적인 연구 가설 또는 연구를 할 수 있는 새로운 방향을 제안하는 질문\n
            
            ---
            
            ### 실험 진행 과정
            
            1. **텍스트 읽기**: 주어진 텍스트를 읽습니다
            2. **질문 작성**: 텍스트에 대한 질문을 작성합니다
            3. **AI 피드백**: AI가 질문의 Bloom 수준을 분류하고 개선 방향을 제안합니다
            4. **설문 응답**: AI 피드백에 대한 간단한 설문에 응답합니다
            5. **질문 수정**: AI 피드백을 참고하여 질문을 수정합니다
            
            이제 2회의 연습을 통해 실험 절차에 익숙해져 보세요.
            """)
            
            if st.button("연습 세션 시작"):
                st.session_state.stage = "practice_ready"
                log_event("Bloom explanation completed")
                st.rerun()
        
        # Handle practice ready stage
        elif st.session_state.stage == "practice_ready":
            st.subheader("연습 세션 안내")
            st.write("이제 2회의 연습을 진행하겠습니다.")
            st.write("연습에서는 실제 실험과 동일한 절차를 따르며, 과정에 익숙해지기 위해 진행합니다. 궁금하신 점이 있으시면 연구자에게 언제든지 질문해주세요!")
            
            if st.button("연습 시작"):
                # Initialize practice session
                st.session_state.practice_mode = True
                st.session_state.iteration = 0
                st.session_state.stage_timers = {}
                st.session_state.current_iteration_data = {}
                
                # Initialize practice paragraphs and balanced condition assignment
                st.session_state.practice_paragraphs = get_practice_paragraphs()
                st.session_state.practice_condition_mapping = create_practice_condition_assignment(
                    st.session_state.participant_id
                )
                
                log_event("Practice session starting", {
                    "practice_condition_mapping": st.session_state.practice_condition_mapping
                })
                
                start_iteration()
                st.rerun()
        
        # Handle practice completion
        elif st.session_state.stage == "practice_completed":
            st.success("연습 세션이 완료되었습니다!")
            
            # Show practice results summary
            if st.session_state.practice_responses:
                st.write("### 연습 세션 요약")
                st.write(f"완료된 연습 반복: {len(st.session_state.practice_responses)}회")
                
                # Experimenter toggle for detailed feedback information
                if st.checkbox("🔬 실험자용 상세 정보 보기", key="experimenter_practice_toggle"):
                    practice_df = pd.DataFrame(st.session_state.practice_responses)
                    
                    # Show feedback types received
                    if 'feedback_type' in practice_df.columns:
                        feedback_types = practice_df['feedback_type'].value_counts()
                        st.write("**받은 피드백 유형:**")
                        for feedback_type, count in feedback_types.items():
                            st.write(f"- {feedback_type}: {count}회")
                    
                    # Show practice condition mapping
                    if hasattr(st.session_state, 'practice_condition_mapping'):
                        st.write("**연습 조건 매핑:**")
                        for iteration, condition in st.session_state.practice_condition_mapping.items():
                            paragraph_idx = PRACTICE_INDICES[iteration] if iteration < len(PRACTICE_INDICES) else "Unknown"
                            st.write(f"- 연습 {iteration + 1} (문단 {paragraph_idx}): {condition}")
                
                # Download option
                practice_csv = get_practice_csv_data()
                if practice_csv:
                    st.download_button(
                        label="연습 세션 결과 다운로드 (CSV)",
                        data=practice_csv,
                        file_name=f"practice_results_{st.session_state.get('participant_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="practice_download"
                    )
            
            st.write("이제 본 실험을 시작합니다.")

            st.markdown("""
            ### 실험 개요
            
            본 실험은 주어진 텍스트를 읽고 질문을 작성한 후, AI로부터 질문에 대한 피드백을 받는 과정으로 구성되어 있습니다. 
            AI는 여러분이 작성한 질문이 Bloom의 분류체계(Bloom's Taxonomy) 중 어느 수준에 해당하는지에 대한 피드백을 제공할 것입니다.
            
            ---
                    
            #### Bloom의 분류체계 6단계:
            
            **1. 기억**: 텍스트 내용을 기억하기 위한 질문\n
            **2. 이해**: 텍스트 내용을 바탕으로 대답할 수 있는 질문; 사실이나 이해, 정의에 대한 질문\n
            **3. 적용**: 텍스트에 대해 추가 내용을 질문하거나(방법, 선행문헌 등) 비슷하지만 다른 상황에 적용하는 질문\n
            **4. 분석**: 텍스트의 내용 요소 간 연결 관계, 인과 관계 등을 묻는 질문, 텍스트 및 저자들의 의도를 묻는 질문\n
            **5. 평가**: (배경지식을 활용하여) 텍스트에 대한 판단 및 비평을 제안하는 질문\n
            **6. 창조**: 창의적인 연구 가설 또는 연구를 할 수 있는 새로운 방향을 제안하는 질문\n
            
            ---
            '본 실험 시작' 버튼을 눌러 실험을 시작해주세요.
            """)
            
            if st.button("본 실험 시작"):
                # Switch to main experiment mode
                st.session_state.practice_mode = False
                st.session_state.practice_completed = True
                st.session_state.iteration = 0
                st.session_state.stage_timers = {}
                st.session_state.current_iteration_data = {}
                
                # Initialize main experiment paragraphs and conditions
                experiment_paragraphs = get_experiment_paragraphs()
                
                # Create randomized paragraph order
                random.shuffle(experiment_paragraphs)
                
                # Store paragraphs
                st.session_state.experiment_paragraphs = experiment_paragraphs
                
                # Create balanced condition assignment
                st.session_state.condition_mapping = create_balanced_condition_assignment(
                    st.session_state.participant_id,
                    experiment_paragraphs
                )
                
                # Log experiment details
                log_event("Main experiment started", {
                    "participant_id": st.session_state.participant_id,
                    "total_paragraphs": len(experiment_paragraphs),
                    "condition_mapping": st.session_state.condition_mapping,
                    "genre_distribution": {genre: sum(1 for p in experiment_paragraphs if p['genre'] == genre) 
                                         for genre in GENRE_RANGES.keys()}
                })
                
                start_iteration()
                st.rerun()
        
        # Display progress and handle experiment stages
        elif st.session_state.stage not in ["practice_completed", "baseline_ready", "bloom_explanation", "practice_ready"]:
            # Show progress information at the top
            if st.session_state.baseline_mode:
                # No progress bar for baseline
                pass
            elif st.session_state.practice_mode:
                progress_bar = st.progress((st.session_state.iteration) / 2)
                st.write(f"연습 {st.session_state.iteration + 1}/2")
            else:
                total_paragraphs = len(st.session_state.experiment_paragraphs)
                progress_bar = st.progress((st.session_state.iteration) / total_paragraphs)
                st.write(f"Iteration {st.session_state.iteration + 1}/{total_paragraphs}")
            
            # Handle different stages
            if st.session_state.stage == "completed":
                st.success("Experiment completed! Thank you for your participation.")
                if hasattr(st.session_state, 'logger'):
                    st.session_state.logger.flush()  # Final flush of all events
                log_files = save_logs()
                if log_files[0]:
                    st.write(f"Event logs saved to: {log_files[0]}")
                if log_files[1]:
                    st.write(f"Response data saved to: {log_files[1]}")
                
                # Display a summary of the responses if needed
                if st.checkbox("Show response summary"):
                    df = pd.DataFrame(st.session_state.responses)
                    st.write(df)
            
            elif st.session_state.stage == "show_paragraph":
                # Display the paragraph
                st.subheader("다음 텍스트를 읽어주세요:")
                if st.session_state.practice_mode:
                    st.write(st.session_state.practice_paragraphs[st.session_state.iteration]['content'])
                else:
                    st.write(st.session_state.experiment_paragraphs[st.session_state.iteration]['content'])
                
                # Wait briefly and then allow moving to the next stage
                time.sleep(0.1)  # Small delay to ensure the UI updates
                
                if st.button("읽기 완료", key="paragraph_read_button"):
                    paragraph_viewed()
            
            elif st.session_state.stage == "ask_question":
                # Show paragraph again as reference
                st.subheader("텍스트:")
                if st.session_state.practice_mode:
                    st.write(st.session_state.practice_paragraphs[st.session_state.iteration]['content'])
                else:
                    st.write(st.session_state.experiment_paragraphs[st.session_state.iteration]['content'])
                
                # Show question input
                st.subheader("텍스트에 대해 떠오르는 질문을 적어주세요:")
                
                user_question = st.text_input(
                    "질문 입력:", 
                    key=f"user_question_{st.session_state.iteration}_{'practice' if st.session_state.practice_mode else 'main'}",
                    on_change=lambda: log_textarea_focus("question_input") if not hasattr(st.session_state, 'question_input_focus_time') else None
                )
                
                # Store question immediately when typed
                st.session_state.user_question = user_question
                
                # Comments section
                question_comments = st.text_area(
                    "[파일럿용 피드백] 질문 입력 관련 코멘트 (선택):",
                    key=f"question_comments_{st.session_state.iteration}_{'practice' if st.session_state.practice_mode else 'main'}",
                    height=100
                )
                
                # Store comments immediately
                st.session_state.question_comments = question_comments
                
                if st.button("질문 제출", key="question_submit_button"):
                    submit_question()
            
            elif st.session_state.stage == "show_feedback":
                # Show paragraph again as reference
                st.subheader("텍스트:")
                if st.session_state.practice_mode:
                    st.write(st.session_state.practice_paragraphs[st.session_state.iteration]['content'])
                else:
                    st.write(st.session_state.experiment_paragraphs[st.session_state.iteration]['content'])
                
                # Show the question
                current_question = st.session_state.current_iteration_data.get('user_question', '')
                
                st.subheader("입력한 질문:")
                st.write(current_question if current_question else "Question not available")
                
                # Show AI feedback
                st.subheader("AI 피드백:")
                st.markdown("""아래는 AI가 연구 참여자의 질문에 대해 제시한 피드백입니다.""")
                current_feedback = st.session_state.current_iteration_data.get('feedback', '')
                st.markdown(f'**{current_feedback}**')
                
                # Comments section
                feedback_comments = st.text_area(
                    "[파일럿용 피드백] AI 피드백 관련 코멘트 (선택):",
                    key=f"feedback_comments_{st.session_state.iteration}_{'practice' if st.session_state.practice_mode else 'main'}",
                    height=100
                )
                
                # Store comments immediately
                st.session_state.feedback_comments = feedback_comments

                # next step explanation
                st.markdown("""다음으로 넘어가면 AI 피드백에 대한 설문이 제시됩니다. AI 피드백을 완전히 숙지하고 넘어가주세요.""")
                
                if st.button("다음", key="feedback_next_button"):
                    send_marker("survey_start")
                    next_stage("survey")
            
            elif st.session_state.stage == "survey":
                # Show survey questions
                st.subheader("다음 설문 문항에 응답해주세요:")
                
                # Always show curiosity question
                curiosity_rating = st.radio(
                    "AI의 피드백에 대해 얼마나 호기심을 느꼈나요?",
                    options=["1", "2", "3", "4", "5", "6", "7"],
                    index=None,
                    key=f"curiosity_{st.session_state.iteration}_{'practice' if st.session_state.practice_mode else 'main'}",
                    help="1 = 전혀 호기심을 느끼지 않음, 7 = 매우 호기심을 느낌",
                    horizontal=True
                )
                
                # Store rating immediately
                if curiosity_rating is not None:
                    st.session_state.curiosity = curiosity_rating
                
                # Always show relatedness question
                relatedness_rating = st.radio(
                    "AI의 피드백이 얼마나 자신의 질문과 관련되었나요?",
                    options=["1", "2", "3", "4", "5", "6", "7"],
                    index=None,
                    key=f"relatedness_{st.session_state.iteration}_{'practice' if st.session_state.practice_mode else 'main'}",
                    help="1 = 전혀 관련되지 않음, 7 = 매우 관련됨",
                    horizontal=True
                )
                
                # Store rating immediately
                if relatedness_rating is not None:
                    st.session_state.relatedness = relatedness_rating

                accept_feedback_option = st.radio(
                    "피드백을 수용할 의향이 있으신가요?",
                    options=["예", "아니오"],
                    index=None,
                    key=f"accept_feedback_{st.session_state.iteration}_{'practice' if st.session_state.practice_mode else 'main'}",
                )
                
                # Store selection immediately
                if accept_feedback_option is not None:
                    st.session_state.accept_feedback = accept_feedback_option
                
                # Comments section
                survey_comments = st.text_area(
                    "[파일럿용 피드백] 설문이나 실험 관련 추가 코멘트 (선택):",
                    key=f"survey_comments_{st.session_state.iteration}_{'practice' if st.session_state.practice_mode else 'main'}",
                    height=100
                )
                
                # Store comments immediately
                st.session_state.survey_comments = survey_comments
                
                if st.button("설문 제출", key="survey_submit_button"):
                    submit_survey()
            
            elif st.session_state.stage == "edit_question":
                # Show the original question and AI suggestion for reference
                current_question = st.session_state.current_iteration_data.get('user_question', '')
                    
                st.subheader("텍스트:")
                if st.session_state.practice_mode:
                    st.write(st.session_state.practice_paragraphs[st.session_state.iteration]['content'])
                else:
                    st.write(st.session_state.experiment_paragraphs[st.session_state.iteration]['content'])
                
                st.subheader("입력한 질문:")
                st.write(current_question if current_question else "Question not available")
                
                st.subheader("AI 피드백:")
                current_feedback = st.session_state.current_iteration_data.get('feedback', '')
                st.markdown(current_feedback)
                
                # Allow editing the question
                st.subheader("질문 수정:")
                st.write("아래 영역에서 질문을 수정할 수 있습니다. 질문을 조금 더 창의적으로 바꾸어보세요.")
                
                # Use current question as initial value
                initial_value = current_question if current_question else ""
                
                edited_question = st.text_area(
                    "수정된 질문:",
                    value=initial_value,
                    key=f"edited_question_{st.session_state.iteration}_{'practice' if st.session_state.practice_mode else 'main'}",
                    height=100
                )
                
                # Store edited question immediately
                st.session_state.edited_question = edited_question
                
                # Comments section
                edit_comments = st.text_area(
                    "[파일럿용 피드백] 질문 수정 관련 코멘트 (선택):",
                    key=f"edit_comments_{st.session_state.iteration}_{'practice' if st.session_state.practice_mode else 'main'}",
                    height=100
                )
                
                # Store comments immediately
                st.session_state.edit_comments = edit_comments

                st.write("최종 제출 버튼을 두 번 누르면 다음으로 넘어갑니다.")
                
                if st.button("최종 제출", key=f"final_submit_button_{st.session_state.iteration}_{'practice' if st.session_state.practice_mode else 'main'}"):
                    submit_edited_question()
    
    # Add download button at the bottom of the screen (outside sidebar)
    # Only show during main experiment, not during practice or baseline
    if (st.session_state.get('started', False) and 
        not st.session_state.get('practice_mode', False) and 
        not st.session_state.get('baseline_mode', False) and 
        st.session_state.get('responses', [])):
        
        st.markdown("---")  # Add a separator line
        st.markdown("### 💾 실험 데이터 다운로드")
        
        csv_data = get_current_csv_data()
        if csv_data:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📥 현재까지 결과 다운로드 (CSV)",
                    data=csv_data,
                    file_name=f"partial_results_{st.session_state.get('participant_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="bottom_download",
                    use_container_width=True
                )
        
        # Also add final download button at completion
        if st.session_state.get('stage') == "completed":
            if 'responses' in st.session_state and st.session_state.responses:
                df = pd.DataFrame(st.session_state.responses)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 최종 실험 결과 다운로드 (CSV)",
                    data=csv,
                    file_name=f"final_results_{st.session_state.participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="final_download",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()
