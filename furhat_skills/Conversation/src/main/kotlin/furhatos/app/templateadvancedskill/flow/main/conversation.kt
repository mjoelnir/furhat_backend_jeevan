package furhatos.app.templateadvancedskill.flow.main

import furhatos.flow.kotlin.*
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
import furhatos.app.templateadvancedskill.params.LOCAL_BACKEND_URL
import furhatos.app.templateadvancedskill.params.AWS_BACKEND_URL
import java.util.concurrent.TimeUnit
import furhatos.app.templateadvancedskill.nlu.UncertainResponseIntent
import java.net.SocketTimeoutException

data class Transcription(val content: String)
data class EngageRequest(val document: String, val answer: String)

/** Helper: choose EN/NO text based on language */
private fun localized(en: String, no: String, lang: AppLanguage): String =
    if (lang == AppLanguage.NO) no else en

/** Helper: use the most recently set conversation language */
private fun currentConversationLanguage(): AppLanguage = LanguageManager.current

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
        furhat.gesture(Gestures.Smile)

        val lang = currentConversationLanguage()
        val intro = localized(
            en = "Hello! I'm here to help you learn about $documentName. What would you like to know?",
            no = "Hei! Jeg er her for å hjelpe deg med $documentName. Hva vil du vite?",
            lang = lang
        )
        furhat.ask(intro)
    }

    onExit {
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
        val userQuestion = it.text.trim()
        conversationCount++

        // Detect language from the *current* question and switch ASR/TTS
        val detectedLanguage = LangDetect.detect(userQuestion)
        setAppLanguage(detectedLanguage)

        val preferredLanguage = when (detectedLanguage) {
            AppLanguage.EN -> "English"
            AppLanguage.NO -> "Norwegian"
        }

        // Mood detection stays as-is
        userMood = when {
            userQuestion.contains(Regex("(great|wonderful|amazing|excellent)", RegexOption.IGNORE_CASE)) -> "positive"
            userQuestion.contains(Regex("(bad|terrible|awful|horrible)", RegexOption.IGNORE_CASE)) -> "negative"
            else -> "neutral"
        }

        if (userQuestion.split(" ").size > 5) {
            furhat.gesture(Gestures.GazeAway, priority = 1)
        }

        // Call backend with preferred_language
        val answer = callDocumentAgent(userQuestion, preferredLanguage)

        val cleanAnswer = answer
            .replace(Regex("https?://\\S+"), "")
            .replace(Regex("\\s+"), " ")
            .trim()

        previousQuestions.add(userQuestion)
        previousAnswers.add(cleanAnswer)
        lastQuestion = userQuestion
        lastAnswer = cleanAnswer

        val currentTime = System.currentTimeMillis()
        if (currentTime - lastGestureTime > 5000) {
            when (userMood) {
                "positive" -> furhat.gesture(Gestures.Smile, priority = 2)
                "negative" -> furhat.gesture(Gestures.ExpressSad, priority = 2)
                else -> furhat.gesture(Gestures.Nod, priority = 2)
            }
            lastGestureTime = currentTime
        }

        furhat.say(cleanAnswer)

        // Localized follow-ups
        val followUpPrompt = when {
            conversationCount == 1 -> when (userMood) {
                "positive" -> localized(
                    en = "What would you like to know more about?",
                    no = "Hva vil du vite mer om?",
                    lang = detectedLanguage
                )
                "negative" -> localized(
                    en = "Would you like me to explain that differently?",
                    no = "Vil du at jeg skal forklare det på en annen måte?",
                    lang = detectedLanguage
                )
                else -> localized(
                    en = "What interests you most about that?",
                    no = "Hva synes du er mest interessant med det?",
                    lang = detectedLanguage
                )
            }

            conversationCount == 2 -> when (userMood) {
                "positive" -> localized(
                    en = "Want to explore that further?",
                    no = "Vil du utforske det videre?",
                    lang = detectedLanguage
                )
                "negative" -> localized(
                    en = "Would you like me to clarify anything?",
                    no = "Vil du at jeg skal forklare noe nærmere?",
                    lang = detectedLanguage
                )
                else -> localized(
                    en = "What would you like to know more about?",
                    no = "Hva vil du vite mer om?",
                    lang = detectedLanguage
                )
            }

            else -> {
                try {
                    val engagePrompt = callEngageUser(documentName, cleanAnswer)
                    if (engagePrompt.isNotEmpty()) {
                        engagePrompt
                    } else {
                        when (userMood) {
                            "positive" -> localized(
                                en = "What would you like to explore next?",
                                no = "Hva vil du utforske videre?",
                                lang = detectedLanguage
                            )
                            "negative" -> localized(
                                en = "Would you like me to explain something else?",
                                no = "Vil du at jeg skal forklare noe annet?",
                                lang = detectedLanguage
                            )
                            else -> localized(
                                en = "What interests you most?",
                                no = "Hva synes du er mest interessant?",
                                lang = detectedLanguage
                            )
                        }
                    }
                } catch (e: Exception) {
                    when (userMood) {
                        "positive" -> localized(
                            en = "What would you like to explore next?",
                            no = "Hva vil du utforske videre?",
                            lang = detectedLanguage
                        )
                        "negative" -> localized(
                            en = "Would you like me to explain something else?",
                            no = "Vil du at jeg skal forklare noe annet?",
                            lang = detectedLanguage
                        )
                        else -> localized(
                            en = "What interests you most?",
                            no = "Hva synes du er mest interessant?",
                            lang = detectedLanguage
                        )
                    }
                }
            }
        }

        when (userMood) {
            "positive" -> furhat.gesture(Gestures.Smile, priority = 2)
            "negative" -> furhat.gesture(Gestures.ExpressSad, priority = 2)
            else -> furhat.gesture(Gestures.Nod, priority = 2)
        }

        furhat.ask(followUpPrompt)
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

// Helper function to call the /ask endpoint (now with preferred_language).
private fun callDocumentAgent(question: String, preferredLanguage: String): String {
    val baseUrl = AWS_BACKEND_URL
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
    val baseUrl = AWS_BACKEND_URL  // or switch based on config if needed
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