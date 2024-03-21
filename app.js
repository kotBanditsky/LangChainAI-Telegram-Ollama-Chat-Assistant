// Import necessary modules
import express from "express";
import "cheerio";
import TelegramBot from "node-telegram-bot-api";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { TOKEN } from "./config.js";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatOllama } from "@langchain/community/chat_models/ollama";

// Create an instance of express
const app = express();

// Create a new instance of OllamaEmbeddings
const embeddings = new OllamaEmbeddings();

// Create a new instance of PromptTemplate with a specific template
const prompt = PromptTemplate.fromTemplate(
  "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Question: {question} Context: {context}"
);

// Create a new instance of ChatOllama
const ollamaLlm = new ChatOllama({
  baseUrl: "http://localhost:11434",
  model: "llama2",
});

// Create a new instance of TelegramBot
const bot = new TelegramBot(TOKEN, {
  polling: true,
});

// Create a var for blocker url/text bot selector check
let blocker = "off";

async function handleUrl(msg, chatId) {
  const url = msg.text;

  // Parse the URL using Cheerio
  const loader = new CheerioWebBaseLoader(url);
  const data = await loader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 20,
  });
  const splitDocs = await textSplitter.splitDocuments(data);
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );

  blocker = "on";
  bot.sendMessage(chatId, "Enter your question:");

  return vectorStore;
}

async function handleText(msg, chatId) {
  const question = msg.text;

  bot.sendMessage(chatId, "Message received! Llama is thinking. handleText", {
    disable_notification: true,
  });

  // Create a stuff documents chain
  const chain = await createStuffDocumentsChain({
    llm: ollamaLlm,
    outputParser: new StringOutputParser(),
    prompt,
  });

  // Invoke the chain with the context and question
  const rest = await chain.invoke({
    context: [],
    question: question,
  });

  bot.sendMessage(chatId, rest, {
    disable_notification: true,
  });
}

async function handleAnswer(msg, chatId, vectorStore) {
  const messageHandler = async (msg) => {
    const question = msg.text;

    bot.sendMessage(
      chatId,
      "Message received! Llama is thinking. handleAnswer",
      {
        disable_notification: true,
      }
    );

    // Search for the most similar document
    const resultOne = await vectorStore.similaritySearch(question, 1);

    // Create a stuff documents chain
    const chain = await createStuffDocumentsChain({
      llm: ollamaLlm,
      outputParser: new StringOutputParser(),
      prompt,
    });

    // Invoke the chain with the context and question
    const rest = await chain.invoke({
      context: resultOne,
      question: question,
    });

    bot.sendMessage(chatId, rest, {
      disable_notification: true,
    });

    blocker = "off";

    // Remove the event listener after processing one message
    bot.off("message", messageHandler);
  };

  bot.on("message", messageHandler);
}

function isUrl(input) {
  try {
    new URL(input);
    return true;
  } catch (err) {
    return false;
  }
}

bot.on("message", async (msg) => {
  const chatId = msg.chat.id;
  if (isUrl(msg.text)) {
    handleUrl(msg, chatId).then(async (vectorStore) =>
      handleAnswer(msg, chatId, vectorStore)
    );
  }
  if (!isUrl(msg.text) && blocker == "off") {
    handleText(msg, chatId);
  }
});

// Start the server
app.listen(PORT, () => console.log(`My server is running on port ${PORT}`));
