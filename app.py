import streamlit as st
import httpx
import io
import csv

st.set_page_config(page_title="SQLGenie - AI-Powered SQL Converter", layout="wide")
st.title("🔮 SQLGenie - AI-Powered Database Assistant")
st.markdown("*Your Database Wish is Our SQL Command*")

# Sidebar settings
with st.sidebar:
	st.header("🔮 SQLGenie Settings")
	backend_url = st.text_input("Backend URL", value="http://127.0.0.1:8000/query")
	st.markdown("**💡 Try These Examples**")
	examples = [
		"List the top 5 most rented films",
		"Find customers from Germany who spent more than $50",
		"Show total revenue by country (top 10)",
		"Which staff processed the most rentals?",
		"How many tables are in the database?",
		"Show me all tables with row counts",
	]
	ex = st.selectbox("Pick an example", options=["(choose)"] + examples)

# Session state for history
if "history" not in st.session_state:
	st.session_state.history = []

# Main input via form (Enter to submit)
st.markdown("### 🗣️ Ask Your Database Question")
with st.form("qform", clear_on_submit=False):
	# Check if there's a suggested query to use
	default_value = ""
	if st.session_state.get("suggested_query"):
		default_value = st.session_state.suggested_query
		# Clear the suggested query after using it
		del st.session_state.suggested_query
	elif ex != "(choose)":
		default_value = ex
	else:
		default_value = st.session_state.get("last_query", "")
	
	user_query = st.text_input("💭 Ask in natural language (e.g., 'Show me the top 5 most rented films')", value=default_value, placeholder="Type your question here...")
	col_run, col_clear = st.columns([1,1])
	with col_run:
		submitted = st.form_submit_button("🔮 Generate SQL", use_container_width=True)
	with col_clear:
		if st.form_submit_button("🗑️ Clear", use_container_width=True):
			st.session_state.last_query = ""
			st.stop()

if submitted and user_query.strip():
	st.session_state.last_query = user_query
	with st.spinner("🔮 SQLGenie is working its magic..."):
		try:
			resp = httpx.post(backend_url, json={"query": user_query}, timeout=60.0)
			try:
				data = resp.json()
			except Exception as json_error:
				# Handle JSON parsing errors
				data = {
					"error": f"Invalid response from server: {str(json_error)}",
					"raw_response": resp.text[:500] if resp.text else "No response text"
				}
			st.session_state.history.insert(0, {"q": user_query, "resp": data})
			if len(st.session_state.history) > 20:
				st.session_state.history.pop()
		except Exception as e:
			# Show technical errors to users
			st.session_state.history.insert(0, {"q": user_query, "resp": {"error": str(e)}})

# Show latest result
if st.session_state.history:
	try:
		latest = st.session_state.history[0]["resp"]
		
		# Show AI correction notice if applicable
		if latest.get("ai_corrected"):
			attempts = latest.get("correction_attempts", 0)
			st.success(f"🤖 AI automatically corrected the query after {attempts} attempt(s)!")
			
			# Show correction history
			if latest.get("correction_history"):
				with st.expander("🔍 View AI Correction History", expanded=False):
					for i, correction in enumerate(latest["correction_history"]):
						try:
							st.markdown(f"**Attempt {correction['attempt']}:**")
							# Show actual technical error message
							error_msg = correction.get('error', 'Unknown error')
							st.error(f"Error: {error_msg}")
							st.markdown("**Original SQL:**")
							st.code(correction.get('original_sql', ''), language="sql")
							st.markdown("**AI Corrected SQL:**")
							st.code(correction.get('corrected_sql', ''), language="sql")
							if correction.get('fallback_used'):
								st.info("Used fallback approach")
							st.markdown("---")
						except Exception as e:
							# If there's any error displaying correction history, skip it
							st.warning("Unable to display correction details for this attempt.")
							continue
			
			# Show original SQL
			with st.expander("Show original SQL"):
				st.code(latest.get("original_sql", ""), language="sql")
		
		# Show legacy correction notice if applicable
		if latest.get("sql_corrected") and not latest.get("ai_corrected"):
			st.warning("⚠️ The SQL query was automatically corrected for better compatibility.")
			with st.expander("Show original SQL"):
				st.code(latest.get("original_sql", ""), language="sql")
		
		# Show fallback notice if applicable
		if latest.get("fallback_used") and not latest.get("ai_corrected"):
			st.info("ℹ️ Used a simplified query approach for better results.")
		
		st.subheader("Generated SQL")
		st.code(latest.get("sql", ""), language="sql")
		
		if "error" in latest:
			st.error(latest.get("error"))
			
			# Show raw response if available (for debugging JSON parsing errors)
			if latest.get("raw_response"):
				with st.expander("🔍 Raw Server Response (for debugging)"):
					st.code(latest.get("raw_response"), language="text")
			
			# Show suggested questions if available
			if latest.get("show_suggestions") and latest.get("suggested_questions"):
				st.markdown("---")
				st.subheader("💡 AI Suggests These Alternative Questions:")
				
				# Parse the suggested questions
				suggestions_text = latest.get("suggested_questions", "")
				suggestions = []
				lines = suggestions_text.split('\n')
				for line in lines:
					line = line.strip()
					if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
						# Remove the number prefix
						question = line[2:].strip()
						if question:
							suggestions.append(question)
				
				# Display suggestions as clickable buttons
				if suggestions:
					st.markdown("**Try asking one of these instead:**")
					for i, suggestion in enumerate(suggestions):
						if st.button(f"💬 {suggestion}", key=f"suggestion_{i}", use_container_width=True):
							# Set the suggested question in the form
							st.session_state.suggested_query = suggestion
							st.rerun()
		else:
			# Show natural language explanation
			if "explanation" in latest:
				st.subheader("📊 Results Explanation")
				st.info(latest.get("explanation"))

			# Show total_tables and row_counts if present
			if "total_tables" in latest:
				st.subheader("Total Number of Tables")
				st.dataframe(latest["total_tables"], use_container_width=True)
			if "row_counts" in latest:
				st.subheader("Row Counts for Each Table")
				st.dataframe(latest["row_counts"], use_container_width=True)

			rows = latest.get("rows", [])
			if rows:
				st.success(f"{len(rows)} row(s)")
				st.dataframe(rows, use_container_width=True)
				# CSV download
				try:
					output = io.StringIO()
					writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
					writer.writeheader()
					writer.writerows(rows)
					st.download_button("📥 Download CSV", data=output.getvalue(), file_name="results.csv", mime="text/csv")
				except Exception as e:
					st.warning("CSV download temporarily unavailable.")
			elif not ("total_tables" in latest or "row_counts" in latest):
				st.info("No rows returned.")
	except Exception as e:
		# If there's any error displaying results, show a friendly message
		st.error("There was an issue displaying the results. Please try again.")

# History accordion
if st.session_state.history:
	with st.expander("History", expanded=False):
		for i, item in enumerate(st.session_state.history):
			st.markdown(f"**Q{i+1}:** {item['q']}")
			resp = item["resp"]
			if resp.get("sql"):
				st.code(resp.get("sql"), language="sql")
			if resp.get("explanation"):
				st.info(f"**Explanation:** {resp.get('explanation')}")
			if resp.get("error"):
				st.error(resp.get("error"))
