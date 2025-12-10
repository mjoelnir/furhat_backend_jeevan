package furhatos.app.templateadvancedskill.trivia

import furhatos.app.templateadvancedskill.params.BACKEND_URL
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.time.Instant
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit

/**
 * Local view of a user's trivia statistics.
 * Cached in-memory and synchronized with the backend via /memory/trivia.
 */
data class TriviaStats(
    var totalQuestions: Int = 0,
    var correctAnswers: Int = 0,
    var lastUpdated: Instant = Instant.now(),
)

/**
 * In-memory cache keyed by userId to reduce backend calls.
 */
private object TriviaStatsCache {
    private val statsByUser = ConcurrentHashMap<String, TriviaStats>()

    fun get(userId: String?): TriviaStats? =
        userId?.let { statsByUser[it] }

    fun replace(userId: String, stats: TriviaStats) {
        statsByUser[userId] = stats
    }

    fun record(userId: String, correct: Boolean): TriviaStats {
        val stats = statsByUser.getOrPut(userId) { TriviaStats() }
        stats.totalQuestions += 1
        if (correct) {
            stats.correctAnswers += 1
        }
        stats.lastUpdated = Instant.now()
        return stats
    }
}

/**
 * Thin HTTP client for trivia stats persistence.
 */
private object TriviaStatsApi {
    private val client: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(10, TimeUnit.SECONDS)
            .writeTimeout(10, TimeUnit.SECONDS)
            .build()
    }

    fun fetch(userId: String): TriviaStats? {
        return try {
            val request = Request.Builder()
                .url("${BACKEND_URL}/memory/trivia/$userId")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return null
                val payload = response.body?.string()?.takeIf { it.isNotBlank() } ?: return null
                parseStats(payload)
            }
        } catch (_: Exception) {
            null
        }
    }

    fun record(userId: String, correct: Boolean): TriviaStats? {
        return try {
            val json = JSONObject()
                .put("user_id", userId)
                .put("correct", correct)

            val request = Request.Builder()
                .url("${BACKEND_URL}/memory/trivia")
                .post(
                    json.toString()
                        .toRequestBody("application/json; charset=utf-8".toMediaType())
                )
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return null
                val payload = response.body?.string()?.takeIf { it.isNotBlank() } ?: return null
                parseStats(payload)
            }
        } catch (_: Exception) {
            null
        }
    }

    private fun parseStats(jsonPayload: String): TriviaStats? {
        return try {
            val json = JSONObject(jsonPayload)
            TriviaStats(
                totalQuestions = json.optInt("total_questions", 0),
                correctAnswers = json.optInt("correct_answers", 0),
                lastUpdated = Instant.now(),
            )
        } catch (_: Exception) {
            null
        }
    }
}

/**
 * Load trivia stats for the given user, using cache first then backend.
 */
fun loadTriviaStats(userId: String?): TriviaStats? {
    if (userId.isNullOrBlank()) return null
    val cached = TriviaStatsCache.get(userId)
    if (cached != null) {
        return cached
    }
    val fetched = TriviaStatsApi.fetch(userId)
    if (fetched != null) {
        TriviaStatsCache.replace(userId, fetched)
    }
    return fetched
}

/**
 * Record a trivia result for the given user.
 * Always updates local cache; attempts to sync to backend and returns the freshest stats.
 */
fun recordTriviaResult(userId: String?, correct: Boolean): TriviaStats? {
    if (userId.isNullOrBlank()) return null
    val local = TriviaStatsCache.record(userId, correct)
    val synced = TriviaStatsApi.record(userId, correct)
    if (synced != null) {
        TriviaStatsCache.replace(userId, synced)
        return synced
    }
    return local
}

