# Replace your main() function with this modified version:

def main():
    st.title("[파일럿] 생성형 AI의 피드백 유형이 질문 수정에 미치는 영향")
    
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
                gender = st.radio("1. 귀하의 성별을 선택하십시오.", ["남", "여"], index=None)
                
                # 2. 나이
                age = st.number_input("2. 귀하의 현재 만 나이를 기입하십시오.", min_value=18, max_value=100, value=None)
                
                # 3. 전공 및 학력 - Using st.data_editor
                st.write("3. 귀하의 전공 및 학력 사항을 모두 기입하십시오.")
                st.write("필요한 경우 행을 추가하여 여러 전공/학력을 입력할 수 있습니다.")
                
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
