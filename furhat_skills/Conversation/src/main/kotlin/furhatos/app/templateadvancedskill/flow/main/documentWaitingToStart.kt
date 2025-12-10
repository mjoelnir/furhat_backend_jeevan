package furhatos.app.templateadvancedskill.flow.main

import furhatos.flow.kotlin.*
import furhatos.app.templateadvancedskill.flow.Parent
import furhatos.app.templateadvancedskill.params.BACKEND_URL
import furhatos.app.templateadvancedskill.perception.UserMemory
import furhatos.app.templateadvancedskill.perception.UserState
import furhatos.app.templateadvancedskill.flow.main.restartPerceptionClient
import furhatos.flow.kotlin.users
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.util.concurrent.TimeUnit

/**
 * Initial handoff state after Greeting.
 * Marks returning users and jumps straight into the trivia flow.
 */
val DocumentWaitingToStart: State = state(parent = Parent) {

    onEntry {
        val profile = UserState.currentProfile

        // Mark the user as seen so future sessions in this process
        // can be treated as returning.
        if (profile != null) {
            UserMemory.markSeen(profile.id)
        }

        // Ensure perception WS is running for every conversation entry.
        restartPerceptionClient(furhat, users.current)

        goto(QuizFromQaPairs)
    }
}

/**
 * Calls the FastAPI /get_docs endpoint with the user input and
 * returns the best matching document name (as provided by the backend).
 */
fun callGetDocs(userInput: String): String? {
    val url = "$BACKEND_URL/get_docs"

    val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    // Build JSON payload.
    val jsonBody = """{"content":"$userInput"}"""
    val body = jsonBody.toRequestBody("application/json".toMediaType())

    val request = Request.Builder()
        .url(url)
        .post(body)
        .build()

    return try {
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException("Unexpected response: $response")
            }
            val respString = response.body?.string() ?: throw IOException("Empty response body")
            val json = JSONObject(respString)
            // Expecting backend to respond with {"response": "<best_document_name>"}
            json.getString("response")
        }
    } catch (e: ConnectException) {
        println("callGetDocs connection error: ${e.message}")
        null
    } catch (e: SocketTimeoutException) {
        println("callGetDocs timeout: ${e.message}")
        null
    } catch (e: Exception) {
        println("callGetDocs general error: ${e.message}")
        null
    }
}

