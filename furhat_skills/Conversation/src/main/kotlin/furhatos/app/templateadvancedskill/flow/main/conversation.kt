package furhatos.app.templateadvancedskill.flow.main

/**
 * Core conversation flow for the trivia experience.
 * Highlights:
 *  - Trivia is driven via backend endpoints (/quiz/question, /trivia/turn).
 *  - Language handling uses LangDetect + explicit pinning via LanguageManager.
 *  - Perception/identity updates are consumed via PerceptionClient (see UserState).
 *  - Trivia stats are cached locally and synced to backend memory endpoints.
 */

import furhatos.flow.kotlin.*
import furhatos.flow.kotlin.users
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.net.ConnectException
import java.io.IOException
import furhatos.nlu.common.*
import furhatos.app.templateadvancedskill.language.AppLanguage
import furhatos.app.templateadvancedskill.language.LangDetect
import furhatos.app.templateadvancedskill.language.LanguageManager
import furhatos.app.templateadvancedskill.language.setAppLanguage
import furhatos.app.templateadvancedskill.flow.Parent
import furhatos.gestures.Gestures
import furhatos.app.templateadvancedskill.params.BACKEND_URL
import java.util.concurrent.TimeUnit
import furhatos.app.templateadvancedskill.nlu.UncertainResponseIntent
import java.net.SocketTimeoutException
import furhatos.app.templateadvancedskill.nlu.MyNameIsIntent
import furhatos.app.templateadvancedskill.perception.UserState
import furhatos.app.templateadvancedskill.perception.UserProfile
import furhatos.app.templateadvancedskill.trivia.TriviaStats
import furhatos.app.templateadvancedskill.trivia.loadTriviaStats
import furhatos.app.templateadvancedskill.trivia.recordTriviaResult


data class Transcription(val content: String)
data class EngageRequest(val document: String, val answer: String)

data class QuizQuestionPayload(val index: Int, val question: String, val answer: String)

/** Helper: choose EN/NO text based on language */
private fun localized(en: String, no: String, lang: AppLanguage): String =
    if (lang == AppLanguage.NO) no else en

/** Helper: use the most recently set conversation language */
private fun currentConversationLanguage(): AppLanguage = LanguageManager.current

/** Helper: Normalize a yes/no style answer. */
private fun matchesKeyword(text: String, keywords: Set<String>): Boolean {
    val normalized = text.trim().lowercase()
        .removeSuffix(".")
        .removeSuffix("!")
        .removeSuffix("?")

    if (normalized.isEmpty()) return false

    return keywords.any { keyword ->
        normalized == keyword ||
            normalized.startsWith("$keyword ") ||
            normalized.endsWith(" $keyword") ||
            normalized.contains(" $keyword ")
    }
}

private fun isAffirmative(text: String): Boolean =
    matchesKeyword(
        text,
        setOf(
            "yes",
            "yeah",
            "yep",
            "yup",
            "sure",
            "ok",
            "okay",
            "ready",
            "of course",
            "absolutely",
            "let's go",
            "lets go",
            "yes please",
            // Norwegian affirmatives
            "ja",
            "javisst",
            "klart",
            "selvfølgelig",
            "jepp",
            "joda",
            "jo",
            "ja takk"
        )
    )

private fun isNegative(text: String): Boolean =
    matchesKeyword(
        text,
        setOf(
            "no",
            "nope",
            "nah",
            "not now",
            "maybe later",
            "another time",
            // Norwegian negatives
            "nei",
            "neida",
            "ikke nå",
            "ikke nå takk",
            "nei takk"
        )
    )

private fun wantsToStop(text: String): Boolean =
    matchesKeyword(
        text,
        setOf(
            "stop",
            "done",
            "finish",
            "enough",
            "thanks",
            "thank you",
            "bye",
            "goodbye",
            "i'm done",
            "im done"
        )
    )

private val norwegianLanguageRequests = listOf(
    "speak norwegian",
    "norwegian please",
    "in norwegian",
    "switch to norwegian",
    "på norsk",
    "snakk norsk",
    "kan du snakke norsk",
    "vennligst norsk"
)

private val englishLanguageRequests = listOf(
    "speak english",
    "english please",
    "in english",
    "switch to english",
    "på engelsk",
    "snakk engelsk",
    "kan du snakke engelsk"
)

private fun explicitLanguageRequest(text: String): AppLanguage? {
    val lower = text.lowercase()
    return when {
        norwegianLanguageRequests.any { lower.contains(it) } -> AppLanguage.NO
        englishLanguageRequests.any { lower.contains(it) } -> AppLanguage.EN
        else -> null
    }
}

private fun languageSwitchAcknowledgement(lang: AppLanguage): String =
    localized(
        en = "Sure, I'll continue in English.",
        no = "Selvfølgelig, jeg fortsetter på norsk.",
        lang = lang
    )

private fun statsProgressLine(stats: TriviaStats, lang: AppLanguage): String =
    localized(
        en = "So far you've answered ${stats.correctAnswers} out of ${stats.totalQuestions} correctly.",
        no = "Så langt har du ${stats.correctAnswers} av ${stats.totalQuestions} riktige.",
        lang = lang
    )

private fun statsFinalLine(stats: TriviaStats, lang: AppLanguage): String =
    localized(
        en = "You finished with ${stats.correctAnswers} correct out of ${stats.totalQuestions}.",
        no = "Du endte med ${stats.correctAnswers} riktige av ${stats.totalQuestions}.",
        lang = lang
    )

private fun namePrompt(lang: AppLanguage): String =
    localized(
        en = "Before we finish, what name should I remember you by so I can keep your score?",
        no = "Før vi avslutter, hvilket navn skal jeg huske deg som for å lagre scoren din?",
        lang = lang
    )

private val namePrefixPatterns = listOf(
    "remember me as",
    "remember me by",
    "call me",
    "my name is",
    "name is",
    "i am",
    "i'm",
    "im",
    "it's",
    "it is",
)

private val rememberMePatterns = listOf(
    "do you remember me",
    "remember me",
    "do you know me",
    "have we met",
    "do you recall me"
)

private val scorePatterns = listOf(
    "what's my score",
    "whats my score",
    "what is my score",
    "how many did i get",
    "how many correct",
    "how am i doing",
    "how did i do",
    "my stats",
    "my score"
)

private fun extractNameFromUtterance(raw: String): String? {
    var candidate = raw.trim()
    if (candidate.isEmpty()) return null

    val lower = candidate.lowercase()
    for (pattern in namePrefixPatterns) {
        val idx = lower.indexOf(pattern)
        if (idx != -1) {
            candidate = candidate.substring(idx + pattern.length).trim()
            break
        }
    }

    candidate = candidate
        .replace(Regex("^(?:the name\\s+)", RegexOption.IGNORE_CASE), "")
        .trim()
        .trim('\'', '"')

    candidate = candidate
        .replace(Regex("[^\\p{L}\\p{M}\\-\\s']"), " ")
        .replace(Regex("\\s+"), " ")
        .trim()

    return candidate.takeIf { it.length >= 2 }
}

private fun isRememberMeQuery(text: String): Boolean {
    val lower = text.lowercase()
    return rememberMePatterns.any { lower.contains(it) }
}

private fun isScoreQuery(text: String): Boolean {
    val lower = text.lowercase()
    return scorePatterns.any { lower.contains(it) }
}

/** Call backend to get a random quiz question from qa_pairs.json. */
private fun fetchQuizQuestion(): QuizQuestionPayload? {
    val baseUrl = BACKEND_URL
    val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    val request = Request.Builder()
        .url("$baseUrl/quiz/question")
        .get()
        .build()

    return try {
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                println("fetchQuizQuestion error: ${response.code} - ${response.message}")
                return null
            }
            val body = response.body?.string() ?: return null
            val json = JSONObject(body)
            val index = json.optInt("index", -1)
            val question = json.optString("question", "")
            val answer = json.optString("answer", "")
            if (question.isBlank() || answer.isBlank()) {
                null
            } else {
                QuizQuestionPayload(index = index, question = question, answer = answer)
            }
        }
    } catch (e: Exception) {
        println("fetchQuizQuestion exception: ${e.message}")
        null
    }
}

private fun prepareTriviaQuestionUtterance(
    lang: AppLanguage,
    preferredLanguage: String
): Pair<QuizQuestionPayload?, String> {
    val quiz = fetchQuizQuestion() ?: run {
        return Pair(
            null,
            localized(
                en = "I couldn't reach the trivia service right now. Please make sure the backend is running and we can try again.",
                no = "Jeg klarte ikke å nå quiz-tjenesten nå. Sørg for at backend kjører, så kan vi prøve igjen.",
                lang = lang
            )
        )
    }

    val llmUtterance = callTriviaTurn(
        phase = "ask",
        question = quiz.question,
        answer = quiz.answer,
        userAnswer = null,
        preferredLanguage = preferredLanguage
    )

    val fallback = localized(
        en = "Here is your question: ${quiz.question}",
        no = "Her er spørsmålet ditt: ${quiz.question}",
        lang = lang
    )

    val toAsk = if (llmUtterance.isNotBlank()) llmUtterance else fallback
    return Pair(quiz, toAsk)
}

/** Call backend /trivia/turn to get a localized trivia utterance from the LLM. */
private fun callTriviaTurn(
    phase: String,
    question: String,
    answer: String,
    userAnswer: String?,
    preferredLanguage: String
): String {
    val baseUrl = BACKEND_URL
    val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    return try {
        val json = JSONObject()
            .put("phase", phase)
            .put("question", question)
            .put("answer", answer)
            .put("preferred_language", preferredLanguage)

        if (userAnswer != null) {
            json.put("user_answer", userAnswer)
        }

        val body = json.toString()
            .toRequestBody("application/json; charset=utf-8".toMediaType())

        val request = Request.Builder()
            .url("$baseUrl/trivia/turn")
            .post(body)
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                return ""
            }
            val respString = response.body?.string() ?: return ""
            val jsonResp = JSONObject(respString)
            return jsonResp.optString("utterance", "")
        }
    } catch (e: Exception) {
        ""
    }
}

// Document Q&A state, inheriting from Parent.
fun documentInfoQnA(documentName: String): State = state(parent = Parent) {

    var conversationCount = 0
    var lastQuestion = ""
    var lastAnswer = ""
    var previousQuestions = mutableListOf<String>()
    var previousAnswers = mutableListOf<String>()
    var userMood = "neutral"
    var lastGestureTime = 0L

    onEntry {
        // For now, the entry into this state is used for the trivia intro flow.
        // We still keep the previous behaviour available if needed elsewhere.
        furhat.gesture(Gestures.Smile)

        val lang = currentConversationLanguage()
        val profile = UserState.currentProfile
        val name = profile?.name

        val greeting = when (lang) {
            AppLanguage.EN -> {
                if (name != null) "Hi $name, would you like to do some trivia?"
                else "Hi there, would you like to do some trivia?"
            }
            AppLanguage.NO -> {
                if (name != null) "Hei $name, har du lyst til å ta en liten quiz?"
                else "Hei, har du lyst til å ta en liten quiz?"
            }
        }

        furhat.ask(greeting)
    }

    onExit {
        // Close perception client when leaving the Q&A state
        users.current.perceptionClient?.close()
        furhat.gesture(Gestures.Wink)
    }

    onResponse<Goodbye> {
        val lang = currentConversationLanguage()
        furhat.gesture(Gestures.Smile)
        furhat.say(
            localized(
                en = "Thank you for the interesting conversation! Goodbye!",
                no = "Takk for en interessant samtale! Ha det bra!",
                lang = lang
            )
        )
        goto(Idle)
    }

    onResponse<No> {
        val lang = currentConversationLanguage()
        if (it.text.matches(Regex("(?i)^(no|nope|nah| no goodbye)$"))) {
            furhat.gesture(Gestures.Nod)
            furhat.say(
                localized(
                    en = "Alright, thank you for the conversation. Goodbye!",
                    no = "Greit, takk for praten. Ha det bra!",
                    lang = lang
                )
            )
            goto(Idle)
        } else {
            raise(it)
        }
    }

    onResponse<MyNameIsIntent> {
        val name = it.intent.name?.toText() ?: return@onResponse

        val profile = UserState.currentProfile
        if (profile != null) {
            profile.name = name
        }

        val client = users.current.perceptionClient
        client?.sendNameUpdate(
            userId = profile?.id,
            name = name
        )

        val lang = currentConversationLanguage()
        furhat.say(
            localized(
                en = "Nice to meet you, $name!",
                no = "Hyggelig å møte deg, $name!",
                lang = lang
            )
        )

        // Go back to the normal questioning flow
        reentry()
    }

    onResponse<UncertainResponseIntent> {
        val lang = currentConversationLanguage()

        furhat.gesture(Gestures.Thoughtful)
        furhat.say {
            random {
                +localized(
                    en = "That's an interesting perspective. Let me share what I know about this topic.",
                    no = "Det er et interessant perspektiv. La meg fortelle det jeg vet om dette temaet.",
                    lang = lang
                )
                +localized(
                    en = "I understand your uncertainty. Let me provide some more information that might help.",
                    no = "Jeg forstår at du er usikker. La meg gi litt mer informasjon som kan hjelpe.",
                    lang = lang
                )
                +localized(
                    en = "That's a good point to explore further. Let me elaborate on this topic.",
                    no = "Det er et godt poeng å utforske videre. La meg utdype dette temaet.",
                    lang = lang
                )
            }
        }

        furhat.ask {
            random {
                +localized(
                    en = "What specific aspect of this topic interests you the most?",
                    no = "Hvilket aspekt ved dette temaet interesserer deg mest?",
                    lang = lang
                )
                +localized(
                    en = "Would you like to explore a particular angle of this discussion?",
                    no = "Vil du utforske en bestemt vinkling av denne diskusjonen?",
                    lang = lang
                )
                +localized(
                    en = "Is there a specific part you'd like me to focus on?",
                    no = "Er det en bestemt del du vil at jeg skal fokusere på?",
                    lang = lang
                )
            }
        }
    }

    onResponse {
        val userText = it.text.trim()

        val explicitRequest = explicitLanguageRequest(userText)
        if (explicitRequest != null) {
            val current = currentConversationLanguage()
            if (current != explicitRequest) {
                setAppLanguage(explicitRequest)
            }
            val lang = explicitRequest
            val ack = languageSwitchAcknowledgement(lang)
            val followUp = when {
                conversationCount == 0 -> localized(
                    en = "Would you like to try a trivia question?",
                    no = "Har du lyst til å prøve et quizspørsmål?",
                    lang = lang
                )
                lastQuestion.isNotBlank() -> localized(
                    en = "Here's the current question again: $lastQuestion",
                    no = "Her er spørsmålet igjen: $lastQuestion",
                    lang = lang
                )
                else -> localized(
                    en = "How would you like to continue?",
                    no = "Hvordan vil du fortsette?",
                    lang = lang
                )
            }
            furhat.ask("$ack $followUp")
            return@onResponse
        }

        // Detect language from the *current* utterance and switch ASR/TTS
        var lang = currentConversationLanguage()
        val detectedLanguage = LangDetect.detect(userText)
        if (detectedLanguage != lang) {
            setAppLanguage(detectedLanguage)
            lang = detectedLanguage
        }

        // FIRST TURN: accept/decline trivia
        if (conversationCount == 0) {
            conversationCount++

            if (isNegative(userText)) {
                val reply = localized(
                    en = "No problem. If you change your mind later, just tell me you want a quiz.",
                    no = "Ikke noe problem. Si fra hvis du vil ha en quiz senere.",
                    lang = lang
                )
                furhat.say(reply)
                goto(Idle)
                return@onResponse
            }

            if (!isAffirmative(userText)) {
                val clarify = localized(
                    en = "Just to be sure – would you like to try one Norwegian trivia question?",
                    no = "Bare så jeg er sikker – vil du prøve et norsk quizspørsmål?",
                    lang = lang
                )
                furhat.ask(clarify)
                return@onResponse
            }

            // User agreed → fetch a question from backend
            val quiz = fetchQuizQuestion()
            if (quiz == null) {
                val errorMsg = localized(
                    en = "I couldn't load a trivia question right now. Let's talk about the documents instead.",
                    no = "Jeg klarte ikke å laste et quizspørsmål nå. La oss heller snakke om dokumentene.",
                    lang = lang
                )
                furhat.say(errorMsg)
                goto(Idle)
                return@onResponse
            }

            lastQuestion = quiz.question
            lastAnswer = quiz.answer

            val intro = localized(
                en = "Great! Here's your question:",
                no = "Supert! Her kommer spørsmålet:",
                lang = lang
            )
            furhat.say(intro)
            furhat.ask(quiz.question)
            return@onResponse
        }

        // SECOND TURN: user answers the quiz question
        if (conversationCount == 1) {
            conversationCount++

            val correct = lastAnswer
            val userAnswer = userText.trim()

            val isCorrect = userAnswer.equals(correct, ignoreCase = true)

            if (isCorrect) {
                val msg = localized(
                    en = "Oh, that was correct! The answer is indeed: $correct.",
                    no = "Det var riktig! Svaret er: $correct.",
                    lang = lang
                )
                furhat.say(msg)
            } else {
                val msg = localized(
                    en = "Nice try, but that wasn't quite right. The correct answer is: $correct.",
                    no = "Godt forsøkt, men det var ikke helt riktig. Det rette svaret er: $correct.",
                    lang = lang
                )
                furhat.say(msg)
            }

            val follow = localized(
                en = "Thanks for playing! If you want to explore the NorwAI documents, just ask me a question.",
                no = "Takk for at du var med! Hvis du vil utforske NorwAI-dokumentene, er det bare å stille et spørsmål.",
                lang = lang
            )
            furhat.say(follow)

            goto(Idle)
            return@onResponse
        }
    }

    onNoResponse {
        val lang = currentConversationLanguage()
        furhat.gesture(Gestures.ExpressSad)
        furhat.ask(
            localized(
                en = "I didn't catch that. Could you please repeat your question?",
                no = "Jeg oppfattet ikke det. Kan du gjenta spørsmålet?",
                lang = lang
            )
        )
        reentry()
    }
}

/**
 * Quiz state driven by QA pairs loaded on the backend.
 *
 * Flow:
 *  - Robot gives a short intro.
 *  - Backend provides a random question–answer pair from qa_pairs.json.
 *  - Robot asks the question and waits for the user(s) to answer.
 *  - Robot tells the user if the answer was correct and reveals the correct answer.
 *  - Robot offers another question.
 */
val QuizFromQaPairs: State = state(parent = Parent) {

    var currentQuestion: String = ""
    var currentAnswer: String = ""
    var expectingAnotherQuestionChoice: Boolean = false
    var awaitingStartConfirmation: Boolean = true
    var awaitingNameCapture: Boolean = false

    onEntry {
        awaitingStartConfirmation = true
        expectingAnotherQuestionChoice = false
        currentQuestion = ""
        currentAnswer = ""
        awaitingNameCapture = false

        val lang = currentConversationLanguage()
        furhat.gesture(Gestures.Smile)
        furhat.ask(
            localized(
                en = "I'm glad you're here. Would you like to try a trivia question?",
                no = "Så hyggelig at du er her. Har du lyst til å prøve et quizspørsmål?",
                lang = lang
            )
        )
    }

    onResponse<Goodbye> {
        val lang = currentConversationLanguage()
        furhat.gesture(Gestures.Smile)

        val profile = UserState.currentProfile
        val userId = profile?.id ?: UserState.getOrCreateTempId()
        val stats = loadTriviaStats(userId)
        stats?.let {
            furhat.say(statsFinalLine(it, lang))
        }

        furhat.say(
            localized(
                en = "Thank you for playing quiz with me. Goodbye!",
                no = "Takk for at du spilte quiz med meg. Ha det bra!",
                lang = lang
            )
        )
        if (profile?.name.isNullOrBlank()) {
            awaitingNameCapture = true
            furhat.ask(namePrompt(lang))
        } else {
            goto(Idle)
        }
    }

    onResponse<MyNameIsIntent> {
        if (!awaitingNameCapture) {
            raise(it)
            return@onResponse
        }

        val lang = currentConversationLanguage()
        val providedName = it.intent.name?.toText()?.trim()
        if (providedName.isNullOrBlank()) {
            furhat.ask(
                localized(
                    en = "I didn't quite catch that name. What should I call you?",
                    no = "Jeg fikk ikke helt med meg navnet. Hva skal jeg kalle deg?",
                    lang = lang
                )
            )
            return@onResponse
        }

        val profile = UserState.currentProfile
        profile?.name = providedName
        val userId = profile?.id ?: UserState.getOrCreateTempId()
        users.current.perceptionClient?.sendNameUpdate(
            userId = userId,
            name = providedName
        )

        awaitingNameCapture = false
        furhat.say(
            localized(
                en = "Great, I'll remember you as $providedName and keep your quiz stats under that name.",
                no = "Flott, jeg skal huske deg som $providedName og lagre quizen din under det navnet.",
                lang = lang
            )
        )
        goto(Idle)
    }

    onResponse {
        val userAnswer = it.text.trim()
        var lang = currentConversationLanguage()

        val explicitRequest = explicitLanguageRequest(userAnswer)
        if (explicitRequest != null) {
            if (lang != explicitRequest) {
                setAppLanguage(explicitRequest)
                lang = explicitRequest
                LanguageManager.pinned = explicitRequest
            }
            val ack = languageSwitchAcknowledgement(lang)
            val followUp = when {
                awaitingStartConfirmation -> localized(
                    en = "Would you like to try a trivia question?",
                    no = "Har du lyst til å prøve et quizspørsmål?",
                    lang = lang
                )
                expectingAnotherQuestionChoice -> localized(
                    en = "Would you like another question, or should we stop here?",
                    no = "Vil du ha et nytt spørsmål, eller skal vi stoppe her?",
                    lang = lang
                )
                currentQuestion.isNotBlank() -> localized(
                    en = "Here is the current question again: $currentQuestion",
                    no = "Her er spørsmålet igjen: $currentQuestion",
                    lang = lang
                )
                else -> localized(
                    en = "How would you like to continue?",
                    no = "Hvordan ønsker du å fortsette?",
                    lang = lang
                )
            }
            furhat.ask("$ack $followUp")
            return@onResponse
        }

        // Handle "do you remember me?" and "what's my score?" at any time.
        val profile = UserState.currentProfile
        val userId = profile?.id ?: UserState.getOrCreateTempId()
        if (isRememberMeQuery(userAnswer)) {
            val knownName = profile?.name?.takeIf { it.isNotBlank() }
            if (knownName != null) {
                val namePart = localized(
                    en = "Yes, $knownName, I remember you.",
                    no = "Ja, $knownName, jeg husker deg.",
                    lang = lang
                )
                furhat.say(namePart)
                val stats = loadTriviaStats(userId)
                if (stats != null && stats.totalQuestions > 0) {
                    furhat.say(statsProgressLine(stats, lang))
                }
                val follow = if (awaitingStartConfirmation) {
                    localized(
                        en = "Would you like to try a trivia question now?",
                        no = "Har du lyst til å prøve et quizspørsmål nå?",
                        lang = lang
                    )
                } else if (currentQuestion.isNotBlank()) {
                    localized(
                        en = "Here is the current question again: $currentQuestion",
                        no = "Her er spørsmålet igjen: $currentQuestion",
                        lang = lang
                    )
                } else {
                    localized(
                        en = "How would you like to continue?",
                        no = "Hvordan vil du fortsette?",
                        lang = lang
                    )
                }
                furhat.ask(follow)
            } else {
                furhat.say(
                    localized(
                        en = "Yes, I recognize you, but I don’t have your name yet.",
                        no = "Ja, jeg kjenner deg igjen, men jeg har ikke navnet ditt ennå.",
                        lang = lang
                    )
                )
                awaitingNameCapture = true
                furhat.ask(namePrompt(lang))
            }
            return@onResponse
        }

        if (isScoreQuery(userAnswer)) {
            val stats = loadTriviaStats(userId)
            if (stats != null && stats.totalQuestions > 0) {
                furhat.say(statsProgressLine(stats, lang))
            } else {
                furhat.say(
                    localized(
                        en = "I haven't recorded any quiz answers for you yet.",
                        no = "Jeg har ikke registrert noen quizsvar for deg ennå.",
                        lang = lang
                    )
                )
            }
            val follow = if (awaitingStartConfirmation) {
                localized(
                    en = "Want to start with a trivia question?",
                    no = "Vil du starte med et quizspørsmål?",
                    lang = lang
                )
            } else if (currentQuestion.isNotBlank()) {
                localized(
                    en = "Here's the current question again: $currentQuestion",
                    no = "Her er spørsmålet igjen: $currentQuestion",
                    lang = lang
                )
            } else {
                localized(
                    en = "Should we continue with another question?",
                    no = "Skal vi fortsette med et nytt spørsmål?",
                    lang = lang
                )
            }
            furhat.ask(follow)
            return@onResponse
        }

        if (awaitingNameCapture) {
            val cleaned = extractNameFromUtterance(userAnswer)

            if (!cleaned.isNullOrBlank()) {
                val profile = UserState.currentProfile
                profile?.name = cleaned
                users.current.perceptionClient?.sendNameUpdate(
                    userId = profile?.id,
                    name = cleaned
                )

                awaitingNameCapture = false
                furhat.say(
                    localized(
                        en = "Perfect, I'll remember you as $cleaned and keep your quiz stats linked to that name.",
                        no = "Supert, jeg skal huske deg som $cleaned og knytte quizstatistikken til det navnet.",
                        lang = lang
                    )
                )
                goto(Idle)
            } else {
                furhat.ask(namePrompt(lang))
            }
            return@onResponse
        }

        val detectedLanguage = LangDetect.detect(userAnswer)
        // Honor pinned language if user explicitly set one; otherwise allow detect to switch on strong signals only.
        val targetLang = LanguageManager.pinned ?: detectedLanguage
        if (targetLang != lang) {
            setAppLanguage(detectedLanguage)
            lang = targetLang
        }

        val preferredLanguage = when (lang) {
            AppLanguage.EN -> "English"
            AppLanguage.NO -> "Norwegian"
        }

        if (awaitingStartConfirmation) {
            when {
                isNegative(userAnswer) -> {
                    furhat.say(
                        localized(
                            en = "No problem, we can do the quiz another time.",
                            no = "Ikke noe problem, vi kan ta quizen en annen gang.",
                            lang = lang
                        )
                    )
                    goto(Idle)
                    return@onResponse
                }

                isAffirmative(userAnswer) -> {
                    awaitingStartConfirmation = false
                    val (quiz, utterance) = prepareTriviaQuestionUtterance(lang, preferredLanguage)
                    if (quiz == null) {
                        furhat.say(utterance)
                        goto(Idle)
                        return@onResponse
                    }

                    currentQuestion = quiz.question
                    currentAnswer = quiz.answer
                    expectingAnotherQuestionChoice = false
                    furhat.ask(utterance)
                    return@onResponse
                }

                else -> {
                    furhat.ask(
                        localized(
                            en = "Just to be sure – would you like to try a trivia question?",
                            no = "Bare så jeg er sikker – vil du prøve et quizspørsmål?",
                            lang = lang
                        )
                    )
                    return@onResponse
                }
            }
        }

        if (expectingAnotherQuestionChoice) {
            when {
                isNegative(userAnswer) ||
                    userAnswer.contains("thanks", ignoreCase = true) ||
                    userAnswer.contains("thank you", ignoreCase = true) ||
                    userAnswer.contains("done", ignoreCase = true) ||
                    userAnswer.contains("stop", ignoreCase = true) -> {
                    furhat.say(
                        localized(
                            en = "Okay, we'll stop the quiz here. Thanks for playing!",
                            no = "Greit, vi avslutter quizen her. Takk for at du spilte!",
                            lang = lang
                        )
                    )
                    val stats = loadTriviaStats(userId)
                    stats?.let { furhat.say(statsFinalLine(it, lang)) }
                    if (profile?.name.isNullOrBlank()) {
                        awaitingNameCapture = true
                        furhat.ask(namePrompt(lang))
                    } else {
                        goto(Idle)
                    }
                    return@onResponse
                }

                isAffirmative(userAnswer) -> {
                    val (quiz, utterance) = prepareTriviaQuestionUtterance(lang, preferredLanguage)
                    if (quiz == null) {
                        furhat.say(utterance)
                        goto(Idle)
                        return@onResponse
                    }

                    currentQuestion = quiz.question
                    currentAnswer = quiz.answer
                    expectingAnotherQuestionChoice = false
                    furhat.ask(utterance)
                    return@onResponse
                }

                else -> {
                    furhat.ask(
                        localized(
                            en = "Just say yes if you want another question, or no if you'd like to stop.",
                            no = "Si bare ja hvis du vil ha et nytt spørsmål, eller nei hvis du vil stoppe.",
                            lang = lang
                        )
                    )
                    return@onResponse
                }
            }
        }

        if (currentQuestion.isBlank()) {
            awaitingStartConfirmation = true
            expectingAnotherQuestionChoice = false
            furhat.ask(
                localized(
                    en = "Let me ask the question first. Shall I start the quiz now?",
                    no = "La meg stille spørsmålet først. Skal jeg starte quizen nå?",
                    lang = lang
                )
            )
            return@onResponse
        }

        if (userAnswer.contains("thanks", ignoreCase = true) ||
            userAnswer.contains("thank you", ignoreCase = true) ||
            userAnswer.contains("done", ignoreCase = true) ||
            userAnswer.contains("stop", ignoreCase = true)
        ) {
            furhat.say(
                localized(
                    en = "No worries, we can stop the quiz here. Thanks for playing!",
                    no = "Ikke noe problem, vi stopper quizen her. Takk for at du spilte!",
                    lang = lang
                )
            )
            val stats = loadTriviaStats(userId)
            stats?.let { furhat.say(statsFinalLine(it, lang)) }
            if (profile?.name.isNullOrBlank()) {
                awaitingNameCapture = true
                furhat.ask(namePrompt(lang))
            } else {
                goto(Idle)
            }
            return@onResponse
        }

        // Basic correctness check for gesture only: case-insensitive full or partial match.
        fun norm(s: String): String = s.lowercase()
            .replace(Regex("[^\\p{L}\\p{N}\\s]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()
        val normUser = norm(userAnswer)
        val normCorrect = norm(currentAnswer)
        val userTokens = normUser.split(" ").filter { it.isNotBlank() }
        val correctTokens = normCorrect.split(" ").filter { it.isNotBlank() }
        val tokenOverlap = if (userTokens.isNotEmpty() && correctTokens.isNotEmpty()) {
            userTokens.intersect(correctTokens.toSet()).size.toDouble() / correctTokens.size.toDouble()
        } else 0.0
        val correct =
            normUser == normCorrect ||
            normCorrect.contains(normUser) ||
            normUser.contains(normCorrect) ||
            tokenOverlap >= 0.4

        if (correct) {
            furhat.gesture(Gestures.Smile, priority = 2)
        } else {
            furhat.gesture(Gestures.ExpressSad, priority = 2)
        }

        recordTriviaResult(userId, correct)

        val llmUtterance = callTriviaTurn(
            phase = "feedback",
            question = currentQuestion,
            answer = currentAnswer,
            userAnswer = userAnswer,
            preferredLanguage = preferredLanguage
        )

        expectingAnotherQuestionChoice = true

        val toAsk = if (llmUtterance.isNotBlank()) {
            llmUtterance
        } else {
            localized(
                en = "The correct answer is: $currentAnswer. Would you like another question?",
                no = "Det riktige svaret er: $currentAnswer. Vil du ha et nytt spørsmål?",
                lang = lang
            )
        }

        // Use ask() so that the user can immediately say yes/no to another question.
        furhat.ask(toAsk)
    }

    onNoResponse {
        val lang = currentConversationLanguage()
        furhat.gesture(Gestures.Nod)
        furhat.ask(
            localized(
                en = "I didn't quite catch your answer. Could you repeat it?",
                no = "Jeg oppfattet ikke helt svaret ditt. Kan du gjenta det?",
                lang = lang
            )
        )
        reentry()
    }
}

// Helper function to call the /ask endpoint (now with preferred_language).
private fun callDocumentAgent(question: String, preferredLanguage: String): String {
    val baseUrl = BACKEND_URL
    val client = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    return try {
        val requestBody = JSONObject()
            .put("content", question)
            .put("preferred_language", preferredLanguage)
            .put("max_tokens", 2000)
            .put("temperature", 0.7)
            .put("top_p", 0.9)
            .toString()
            .toRequestBody("application/json; charset=utf-8".toMediaType())

        val request = Request.Builder()
            .url("$baseUrl/ask")
            .post(requestBody)
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                println("Error response from backend: ${response.code} - ${response.message}")
                throw IOException("Unexpected response: $response")
            }

            val jsonResponse = response.body?.string() ?: throw IOException("Empty response")
            val jsonObject = JSONObject(jsonResponse)

            val responseText = jsonObject.getString("response")
            println("Response length: ${responseText.length} characters")

            if (responseText.endsWith("...") ||
                !responseText.endsWith(".") ||
                responseText.length > 1900
            ) {
                println("Warning: Response might be truncated")
            }

            responseText
        }

    } catch (e: ConnectException) {
        val lang = currentConversationLanguage()
        localized(
            en = "I'm sorry, I cannot process your request right now. Please try again in a moment.",
            no = "Beklager, jeg kan ikke behandle forespørselen din akkurat nå. Prøv igjen om litt.",
            lang = lang
        )
    } catch (e: SocketTimeoutException) {
        val lang = currentConversationLanguage()
        localized(
            en = "I'm sorry, the request took too long to process. Please try asking your question again.",
            no = "Beklager, forespørselen tok for lang tid. Prøv å stille spørsmålet på nytt.",
            lang = lang
        )
    } catch (e: Exception) {
        val lang = currentConversationLanguage()
        localized(
            en = "I apologize, but I encountered an error processing your question. Could you please rephrase it?",
            no = "Beklager, det oppstod en feil da jeg skulle behandle spørsmålet ditt. Kan du formulere det på en annen måte?",
            lang = lang
        )
    }
}

// Helper function to call the /engage endpoint (unchanged logic, language handled by backend).
private fun callEngageUser(documentName: String, answer: String): String {
    val baseUrl = BACKEND_URL  // automatically resolves to local if available
    val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    return try {
        val map = JSONObject()
        map.put("document", documentName)
        map.put("answer", answer)

        val requestBody = map.toString()
            .toRequestBody("application/json; charset=utf-8".toMediaType())

        val request = Request.Builder()
            .url("$baseUrl/engage")
            .post(requestBody)
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw IOException("Unexpected response: $response")
            val jsonResponse = response.body?.string() ?: throw IOException("Empty response")
            val jsonObject = JSONObject(jsonResponse)
            jsonObject.optString("prompt", "")
        }
    } catch (e: ConnectException) {
        ""
    } catch (e: Exception) {
        ""
    }
}