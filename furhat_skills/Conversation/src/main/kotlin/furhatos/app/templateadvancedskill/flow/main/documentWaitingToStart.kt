package furhatos.app.templateadvancedskill.flow.main

import furhatos.flow.kotlin.*
import furhatos.app.templateadvancedskill.flow.Parent
import furhatos.app.templateadvancedskill.params.AWS_BACKEND_URL
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.util.concurrent.TimeUnit

val DocumentWaitingToStart: State = state(parent = Parent) {

    onEntry {
        furhat.ask(
            "I'm ready to assist with your document questions. " +
            "Could you please tell me what subject you're interested in, " +
            "or simply the name of the document?"
        )
    }

    // When any response is detected, transition to document-specific Q&A.
    onResponse {
        val userInput = it.text.trim()

        // Call the API endpoint /get_docs to perform document retrieval/classification.
        val bestDocName = callGetDocs(userInput)

        if (bestDocName.isNullOrBlank()) {
            furhat.say(
                "I'm having trouble finding a matching document right now. " +
                "Could you try rephrasing the title or subject?"
            )
            reentry()
        } else {
            goto(documentInfoQnA(bestDocName))
        }
    }

    onNoResponse {
        furhat.ask(
            "I didn't catch that. Please tell me the subject or the name of the document you're interested in."
        )
        reentry()
    }
}

/**
 * Calls the FastAPI /get_docs endpoint with the user input and
 * returns the best matching document name (as provided by the backend).
 */
fun callGetDocs(userInput: String): String? {
    val url = "$AWS_BACKEND_URL/get_docs"

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
        // Fallback value – you may want to handle this more gracefully in your flow
        "I'm sorry, I cannot connect to the server right now. Please try again later."
    } catch (e: SocketTimeoutException) {
        "I'm sorry, the server is taking too long to respond. Please try again later."
    } catch (e: Exception) {
        "I apologize, but I encountered an error processing your request."
    }
}

